import os
import mne
from pathlib import Path
import pylossless as ll
import matplotlib.pyplot as plt
import threading
import time
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
import numpy as np

mne.viz.set_browser_backend('qt')

def load_lossless_derivative(edf_path, do_clean=True):
    """
    Load a Lossless derivative file and check for a corresponding ll_config file in the same directory.
    
    Parameters:
    -----------
    edf_path : str or Path
        Path to the Lossless derivative file
    do_clean : bool, optional
        Whether to perform cleaning operations on the data. Defaults to True.
        
    Returns:
    --------
    ll_state : ll.LosslessPipeline
        The loaded Lossless derivative state
    """

    # Convert to Path object
    edf_path = Path(edf_path)
    
    # Check if EDF exists
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    
    # Look for ll_config file in the same directory
    parent_dir = edf_path.parent
    config_file = None
    
    # Search for any file containing 'll_config' in its name
    for file in parent_dir.glob('*ll_config*'):
        config_file = file
        print(f"Found config file: {file}")
        break
    
    if config_file is None:
        raise FileNotFoundError(f"No ll_config file found in directory: {parent_dir}")
    
    # Load the EDF file
    try:
        ll_state = ll.LosslessPipeline()
        ll_state = ll_state.load_ll_derivative(edf_path)
        print(f"Successfully loaded EDF file: {edf_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading EDF file: {str(e)}")
    
    # Get bad channels using the correct flags accessor and add them to raw.info['bads']
    bad_channels = ll_state.flags['ch'].get_flagged()
    ll_state.raw.info['bads'] = bad_channels
    print("\nBad channels:")
    print(bad_channels)

    # Get IC flags and group by IC type
    ic_flags = ll_state.flags['ic']
    ic_counts = ic_flags.groupby('ic_type').size()
    print("\nIC type counts:")
    print(ic_counts)

    # Add flagged components to exclude list
    artifact_types = ['eog', 'ecg', 'muscle', 'line_noise', 'channel_noise']
    flagged_components = ic_flags[ic_flags['ic_type'].isin(artifact_types)].index.tolist()
    ll_state.ica2.exclude.extend(flagged_components)
    print(f"\nExcluded {len(flagged_components)} components of types: {artifact_types}")
    
    if do_clean:
        print("Starting cleaning")
        # Create a copy of the raw data and apply all cleaning steps in sequence
        cleaned_state = (ll_state.raw
                        .copy()
                        .load_data()  # Load data into memory
                        .set_annotations(ll_state.raw.annotations))  # Preserve annotations
        
        # Apply ICA to remove artifacts
        ll_state.ica2.apply(cleaned_state)
        cleaned_state = cleaned_state.set_eeg_reference('average')
        
        # Clean up bad channels and filter
        cleaned_state = (cleaned_state
                        .interpolate_bads()  # Fix bad channels
                        .filter(l_freq=1.0, h_freq=30.0))  # Apply bandpass filter
        # Apply average reference
        cleaned_state = cleaned_state.set_eeg_reference('average')
        
        return cleaned_state
    else:
        print("Skipping cleaning")
        return ll_state

def plot_all_ic_topos(ll_state):
    """
    Plot topographical maps for all ICs from a Lossless state.
    Creates and monitors a .local_reject file for bad components.
    Clicking on any IC will open it in a new figure window.
    Text will be shown in red for components marked as artifacts.
    """
    from PyQt5.QtCore import QTimer, QObject, pyqtSignal
    
    class FileWatcher(QObject):
        file_changed = pyqtSignal()
        
        def __init__(self, file_path):
            super().__init__()
            self.file_path = file_path
            self.last_modified = file_path.stat().st_mtime
            
        def check_file(self):
            try:
                current_modified = self.file_path.stat().st_mtime
                if current_modified > self.last_modified:
                    self.last_modified = current_modified
                    self.file_changed.emit()
            except Exception as e:
                print(f"Error monitoring file: {e}")

    # Get the number of ICs
    n_components = ll_state.ica2.n_components_
    
    # Create initial file with known bad components
    bad_components = set()
    if hasattr(ll_state, 'flags') and 'ic' in ll_state.flags:
        ic_flags = ll_state.flags['ic']
        artifact_types = ['eog', 'ecg', 'muscle', 'line_noise', 'channel_noise']
        bad_components = {f'ICA{str(x).zfill(3)}' for x in ic_flags[ic_flags['ic_type'].isin(artifact_types)].index}
    
    local_reject_file = Path('.local_reject')
    with open(local_reject_file, 'w') as f:
        f.write("0,0\n")  # First line is time range
        f.write(str(bad_components))  # Second line is bad components
    
    # Function to read bad components and time range from file
    def get_bad_components():
        with open(local_reject_file, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return 0, 0, set()
            
            # Get time range from first line
            try:
                xmin, xmax = map(float, lines[0].strip().split(','))
            except:
                xmin, xmax = 0, 0
                
            # Get bad components from second line
            if not lines[1].strip():
                return xmin, xmax, set()
            
            # Remove set formatting and split into components from second line
            components = lines[1].strip('{}').replace("'", "").split(', ')
            return xmin, xmax, {comp for comp in components if comp}

    # Create new figure with subplots
    n_cols = 5
    n_rows = (n_components + 4) // 5
    
    fig = ll_state.ica2.plot_components(
        picks=range(n_components),
        ch_type='eeg',
        title='IC Topographies',
        show=False,
        ncols=n_cols,
        nrows=n_rows,
    )
    
    fig.set_size_inches(15, 3 * n_rows)
    
    # Get current bad components and convert to indices
    _, _, bad_components = get_bad_components()
    bad_indices = {int(comp.replace('ICA', '')) for comp in bad_components}
    
    # Update component labels
    if hasattr(ll_state, 'flags') and 'ic' in ll_state.flags:
        ic_flags = ll_state.flags['ic']
        
        for idx in range(n_components):
            ax = fig.axes[idx]
            ax.set_title('')
            
            if idx in ic_flags.index:
                ic_type = ic_flags.loc[idx, 'ic_type']
                confidence = ic_flags.loc[idx, 'confidence']
                # Check if this component is in the bad list
                text_color = 'red' if idx in bad_indices else 'black'
                ax.text(0.5, -0.1, f'IC{idx}\n{ic_type}\n{confidence:.2f}', 
                        horizontalalignment='center',
                        verticalalignment='top',
                        transform=ax.transAxes,
                        fontsize=8,
                        color=text_color)
    
    plt.subplots_adjust(bottom=0.1, hspace=0.25, wspace=0.3)
    fig.canvas.draw_idle()
    plt.show(block=False)

    def foo():
        if not hasattr(foo, '_called'):
            foo._called = True
            return
            
        xmin, xmax, bad_components = get_bad_components()
        
        # Create a new figure for scalp plot with increased height
        plt.figure(figsize=(12, 20))
        
        # Extract data for the time window
        data, times = ll_state.raw[:, int(xmin * ll_state.raw.info['sfreq']):int(xmax * ll_state.raw.info['sfreq'])]
        
        # Normalize each channel and apply large offset
        n_channels = data.shape[0]
        for i, ch_name in enumerate(ll_state.raw.ch_names):
            # Normalize the data to [-1, 1] range
            channel_data = data[i]
            normalized_data = channel_data / (np.max(np.abs(channel_data)) + 1e-6)
            # Apply large offset between channels
            plt.plot(times, normalized_data + (n_channels - i) * 3, label=ch_name, linewidth=0.8)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Channels')
        plt.title('Scalp Data Snapshot')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        return

    # Set up Qt-based file monitoring
    watcher = FileWatcher(local_reject_file)
    watcher.file_changed.connect(foo)
    
    # Create timer in the main Qt thread
    timer = QTimer()
    timer.timeout.connect(watcher.check_file)
    timer.start(500)  # Check every 500ms
    
    # Store timer and watcher as figure properties to prevent garbage collection
    fig.timer = timer
    fig.watcher = watcher
    
    # Define click event handler
    def on_click(event):
        if event.inaxes:
            ax_idx = fig.axes.index(event.inaxes)
            if ax_idx < n_components:
                new_fig = ll_state.ica2.plot_components(
                    picks=[ax_idx],
                    ch_type='eeg',
                    show=False
                )
                if hasattr(ll_state, 'flags') and 'ic' in ll_state.flags:
                    if ax_idx in ic_flags.index:
                        ic_type = ic_flags.loc[ax_idx, 'ic_type']
                        confidence = ic_flags.loc[ax_idx, 'confidence']
                        new_fig.suptitle(f'IC{ax_idx} ({ic_type})\nconfidence: {confidence:.2f}')
                plt.show()

    fig.canvas.mpl_connect('button_press_event', on_click)
    return fig

def plot_ic_scrollplot(ll_state, picks=None):
    """
    Plot the scrolling time course of Independent Components (ICs).
    
    Parameters:
    -----------
    ll_state : ll.LosslessPipeline
        The loaded Lossless derivative state containing ICA information
    picks : list or None, optional
        List of component indices to plot. If None, plots all components.
    """
    # If no specific components are selected, plot all
    if picks is None:
        picks = range(ll_state.ica2.n_components_)
    
    # Plot scrolling time course with optimizations
    ll_state.ica2.plot_sources(ll_state.raw.crop(tmin=0, tmax=10), picks=picks,
                              start=0, show=True,
                              title='IC Time Courses',
                              block=True)
    return

if __name__ == "__main__":
    edf_path = "subjects/sub-hcS01/ses-S1/eeg/sub-hcS01_ses-S1_task-pyl_eeg.edf"
    ll_state = load_lossless_derivative(edf_path, do_clean=False)
    
    # Plot both visualizations
    fig = plot_all_ic_topos(ll_state)
    plot_ic_scrollplot(ll_state)
    
    print('Done')