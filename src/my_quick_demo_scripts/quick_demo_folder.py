from  pathlib import Path
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

folder_name = "mcgs_fav_imgs"
input_folder = Path(f"examples/single_img_data/{folder_name}")
output_folder = Path(f"examples/single_img_results/{folder_name}")
prompt = "remove degradation"

if not input_folder.exists():
    print(f"input folder not found at path: {input_folder}")
    exit(1)

output_folder.mkdir(parents=True, exist_ok=True)

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

for input_path in input_folder.glob("*.*"):
    output_path = output_folder / input_path.name
    input_image = load_image(str(input_path))
    output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
    output_image.save(str(output_path))