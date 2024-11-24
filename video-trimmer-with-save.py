import cv2
import numpy as np
from datetime import datetime
import os
from typing import List, Tuple, Union

class VideoFrameExtractor:
    def __init__(self, video_path: str):
        """
        Initialize the video frame extractor
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")
            
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
    
    def time_to_frame(self, time_str: str) -> int:
        """Convert time string to frame number"""
        if isinstance(time_str, (int, float)):
            seconds = float(time_str)
        else:
            try:
                time_obj = datetime.strptime(time_str, '%H:%M:%S')
                seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
            except ValueError:
                try:
                    time_obj = datetime.strptime(time_str, '%M:%S')
                    seconds = time_obj.minute * 60 + time_obj.second
                except ValueError:
                    raise ValueError("Time must be in 'HH:MM:SS' or 'MM:SS' format, or seconds as number")
        
        frame_number = int(seconds * self.fps)
        return frame_number
    
    def extract_and_save(self, 
                        start_time: Union[str, float], 
                        num_frames: int = 15,
                        output_path: str = None,
                        save_frames: bool = False,
                        frame_dir: str = None) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract frames and optionally save as video and/or individual frames
        
        Args:
            start_time: Start time in 'HH:MM:SS' or 'MM:SS' format, or seconds as number
            num_frames: Number of frames to extract
            output_path: Path to save output video file
            save_frames: Whether to save individual frames
            frame_dir: Directory to save individual frames (if save_frames is True)
            
        Returns:
            Tuple of (list of frames, list of frame timestamps)
        """
        start_frame = self.time_to_frame(start_time)
        
        if start_frame + num_frames > self.frame_count:
            raise ValueError(f"Not enough frames in video after {start_time}")
        
        # Set video position to start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        frame_times = []
        
        # Create video writer if output_path is provided
        video_writer = None
        if output_path:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Try to use the same codec as the input video
            try:
                video_writer = cv2.VideoWriter(
                    output_path,
                    self.original_fourcc,
                    self.fps,
                    (self.frame_width, self.frame_height)
                )
            except:
                # Fallback to mp4v codec if original codec doesn't work
                video_writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    self.fps,
                    (self.frame_width, self.frame_height)
                )
        
        # Create frame directory if saving individual frames
        if save_frames and frame_dir:
            os.makedirs(frame_dir, exist_ok=True)
        
        # Extract frames
        for i in range(num_frames):
            ret, frame = self.cap.read()
            
            if not ret:
                break
                
            frames.append(frame)
            frame_times.append(self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            
            # Write frame to video file
            if video_writer is not None:
                video_writer.write(frame)
            
            # Save individual frame if requested
            if save_frames and frame_dir:
                frame_path = os.path.join(frame_dir, f"frame_{i:03d}.jpg")
                cv2.imwrite(frame_path, frame)
        
        # Clean up video writer
        if video_writer is not None:
            video_writer.release()
        
        return frames, frame_times
    
    def __del__(self):
        """Clean up video capture object"""
        if hasattr(self, 'cap'):
            self.cap.release()

def process_video_segment(
    video_path: str,
    start_time: Union[str, float],
    output_video_path: str = "output_segment.mp4",
    num_frames: int = 15,
    save_individual_frames: bool = False,
    frame_dir: str = "extracted_frames"
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Process video segment and save as video and/or individual frames
    
    Args:
        video_path: Path to input video file
        start_time: Start time in 'HH:MM:SS' or 'MM:SS' format, or seconds as number
        output_video_path: Path to save output video file
        num_frames: Number of frames to extract
        save_individual_frames: Whether to save individual frames
        frame_dir: Directory to save individual frames
        
    Returns:
        Tuple of (list of frames, list of frame timestamps)
    """
    try:
        # Initialize extractor
        extractor = VideoFrameExtractor(video_path)
        
        # Extract and save frames
        frames, frame_times = extractor.extract_and_save(
            start_time=start_time,
            num_frames=num_frames,
            output_path=output_video_path,
            save_frames=save_individual_frames,
            frame_dir=frame_dir
        )
        
        print(f"Successfully extracted {len(frames)} frames")
        print(f"Frame timestamps: {[f'{t:.3f}s' for t in frame_times]}")
        print(f"Video saved to: {output_video_path}")
        
        if save_individual_frames:
            print(f"Individual frames saved to: {frame_dir}")
        
        # Display video information
        print("\nVideo Information:")
        print(f"FPS: {extractor.fps}")
        print(f"Resolution: {extractor.frame_width}x{extractor.frame_height}")
        print(f"Segment duration: {frame_times[-1] - frame_times[0]:.3f} seconds")
            
        return frames, frame_times
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return [], []

# Example usage
if __name__ == "__main__":
    # Example parameters
    video_path = "/home/deepb/Desktop/AIDS/project/Zebra Tries to Kill Foal While Mother Fights Back - Latest Sightings (1080p, h264, youtube).mp4"
    start_time = "00:00:37"  # or use seconds like 90.0
    output_video = "output/trimmed_segment.mp4"
    frame_directory = "output/frames"
    
    frames, timestamps = process_video_segment(
        video_path=video_path,
        start_time=start_time,
        output_video_path=output_video,
        num_frames=30,
        save_individual_frames=True,
        frame_dir=frame_directory
    )
