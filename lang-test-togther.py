from together import Together
import base64
from io import BytesIO
from PIL import Image
import random
import time
import io
import os
from typing import Dict, Any
from langgraph.graph import Graph



def generate_random_seed(state: Dict[str, Any]) -> Dict[str, Any]:
    current_time = time.time_ns()
    random.seed(current_time)
    state['seed'] = random.randint(0, 2**32 - 1)
    return state

def prepare_prompt(state: Dict[str, Any]) -> Dict[str, Any]:

    user_input = state.get('user_input', {})
    prompts = {
        "world": user_input.get("world", "A fantasy forest with a castle"),
        "characters": user_input.get("characters", ["a brave knight", "a fierce dragon"]),
        "scene": user_input.get("scene", "a knight battles a dragon in the forest."),
    }
    prompts["seed"] = state['seed']
    state['prompts'] = prompts
    return state

def llama_generate(messages):
    client = Together()
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        messages=messages,
        max_tokens=None,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>","<|eom_id|>"],
        stream=False
    )
    return response.choices[0].message.content

def generate_prompt(state: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = state['prompts']
    messages = [
        {
            "role": "system",
            "content": """Create a prompt for a text-to-image model to generate a 
            description for a video that will be used to create a story with the following: 
            characters: {characters}
            scene: {scene}
            world: {world}
            """.format(characters=user_prompt["characters"], scene=user_prompt["scene"], world=user_prompt["world"])
        }
    ]
    description = llama_generate(messages)
    
    messages = [
        {
            "role": "system",
            "content": """Create a shot division for a video based on the following description.
            description: {description}
            """.format(description=description)
        }
    ]
    shot_divisions = llama_generate(messages)
    print(shot_divisions)
    
    state['shot_divisions'] = shot_divisions
    return state

def parse_shot_divisions(state: Dict[str, Any]) -> Dict[str, Any]:
    shot_divisions = state['shot_divisions']
    shots = {}
    current_shot = None
    current_description = []
    for line in shot_divisions.split('\n'):
        line = line.strip()
        if line.startswith("**Shot"):
            if current_shot:
                shots[current_shot] = ' '.join(current_description)
            current_shot = line.split(':')[0].strip('*').strip().split()[1]
            current_description = []
        elif line.startswith("*") and current_shot:
            current_description.append(line.lstrip('* '))
    if current_shot:
        shots[current_shot] = ' '.join(current_description)
    state['parsed_shot_divisions'] = shots
    return state

def generate_image(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = state['parsed_shot_divisions']
    seed = state['seed']
    images_b64_dict = {}
    client = Together()

    # generate more images per shot by changing n =1 and making  images_b64_dict[shot_number] a list
    for shot_number, description in prompt.items():
        print(f"generating image for Shot {shot_number} with description: {description}")
        response = client.images.generate(
            prompt=f"generate an image frame for a video with the following description: {description}",
            model="black-forest-labs/FLUX.1-schnell",
            width=1024,
            height=768,
            steps=4,
            n=1,
            seed=seed,
            response_format="b64_json"
        )
        images_b64_dict[shot_number] = response.data[0].b64_json

    state['images_b64_dict'] = images_b64_dict
    return state

def process_images(state: Dict[str, Any]) -> Dict[str, Any]:
    images_b64_dict = state['images_b64_dict']
    seed = state['seed']
    directory = f"generated_images_{seed}"
    os.makedirs(directory, exist_ok=True)

    for shot_number, b64_image in images_b64_dict.items():
        image_data = base64.b64decode(b64_image)
        image = Image.open(io.BytesIO(image_data))
        image.show()
        filename = os.path.join(directory, f"shot_{shot_number}.png")
        image.save(filename)
        print(f"Saved image for Shot {shot_number} as {filename}")

    state['output_directory'] = directory
    return state

workflow = Graph()

def main():

    workflow.add_node("generate_random_seed", generate_random_seed)
    workflow.add_node("prepare_prompt", prepare_prompt)
    workflow.add_node("generate_prompt", generate_prompt)
    workflow.add_node("parse_shot_divisions", parse_shot_divisions)
    workflow.add_node("generate_image", generate_image)
    workflow.add_node("process_images", process_images)

    workflow.add_edge("generate_random_seed", "prepare_prompt")
    workflow.add_edge("prepare_prompt", "generate_prompt")
    workflow.add_edge("generate_prompt", "parse_shot_divisions")
    workflow.add_edge("parse_shot_divisions", "generate_image")
    workflow.add_edge("generate_image", "process_images")

    workflow.set_entry_point("generate_random_seed")

    app = workflow.compile()

    result = app.invoke({})
    print(f"Images saved in directory: {result['output_directory']}")

    prompts = {
    "world": "A fantasy forest with a castle",
    "characters": ["a brave knight", "a fierce dragon"],
    "scene": "a knight battles a dragon in the forest.",
    }

    app = workflow.compile()


    result = app.invoke({"user_input": prompts})

if __name__ == "__main__":
    main()
