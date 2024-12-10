# OmniAnimate Client

## Usage

### Command Line Interface

#### Synchronous Mode (Wait for completion)

```bash
python client.py \
    --ref-image path/to/image.jpg \
    --drive-video path/to/video.mp4 \
    --token omni-ae342931-814d-4a49-97a6-cedf6af3dd18 \
    --server http://129.213.81.69:6006 \
    --height 1024 \
    --width 576
```

#### Asynchronous Mode (Returns task ID immediately)

```bash
python client.py \
    --ref-image path/to/image.jpg \
    --token omni-ae342931-814d-4a49-97a6-cedf6af3dd18 \
    --server http://129.213.81.69:6006 \
    --async
```

### Python API Usage

#### Synchronous Call Example

```python
from client import OmniAnimateClient

# Create client instance
client = OmniAnimateClient("http://localhost:8000")

# Synchronous call (wait for completion)
output_path = client.create_animation(
    ref_image_path="path/to/image.jpg",
    drive_video_path="path/to/video.mp4",
    token="omni-ae342931-814d-4a49-97a6-cedf6af3dd18",
    height=1024,
    width=576
)
print(f"Output video path: {output_path}")
```

#### Asynchronous Call Example

```python
# Asynchronous call
task_id = client.create_animation(
    ref_image_path="path/to/image.jpg",
    drive_video_path="path/to/video.mp4",
    token="omni-ae342931-814d-4a49-97a6-cedf6af3dd18",
    wait_complete=False
)
print(f"Task ID: {task_id}")

# Check task status
status = client.get_task_status(task_id)
print(f"Task status: {status}")

# Once the task is complete, write the video to disk
if status['status'] == 'completed':
    output_path = client.get_output_video(task_id, save_dir="./results")
    print(f"Video saved to: {output_path}")
```

## API Parameters

### OmniAnimateClient.create_animation()

Parameter details:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| ref_image_path | str | Required | Path to reference image |
| drive_video_path | str | Required | Path to driving video |
| token | str | Required | API authentication token |
| height | int | 1024 | Output video height |
| width | int | 576 | Output video width |
| keep_ratio | bool | True | Maintain aspect ratio |
| keep_ref_dim | bool | False | Keep reference image dimensions |
| stride | int | 1 | Video frame sampling stride |
| steps | int | 20 | Number of inference steps |
| seed | int | 1234 | Random seed |
| guidance_scale | float | 3.0 | Guidance scale factor |

### Return Values

- Synchronous mode (wait_complete=True): Returns output video path
- Asynchronous mode (wait_complete=False): Returns task ID

## Important Notes

1. Default dimensions: height=1024, width=576 (for upper body/portrait). For full-body animations, set width to 768 .
2. Reference image and driving video files must exist and be in correct formats
3. Asynchronous mode requires manual task status monitoring
4. Output files are saved in the `./results` directory by default