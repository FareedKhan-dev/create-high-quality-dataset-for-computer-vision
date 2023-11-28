# Computer Vision Project: Generating Realistic Datasets with ChatGPT

Computer vision projects often require high-quality datasets to train robust models. In this blog post, we'll guide you through a simple process of generating a diverse and realistic dataset using a one-click approach. By leveraging the power of ChatGPT and a realistic vision image generation model, you can easily create a dataset tailored to your specific needs.

The code for this blog can be found at the following GitHub repository: [GitHub link](GitHUB link)

## Step 1: Generating Realistic Image Prompts Using ChatGPT

In this initial step, we use the below ChatGPT free prompt to dynamically generate prompts for realistic-vision images. To tailor the prompts to your specific needs, you can adjust the parameters in the provided Python code.

```python
# Parameters
important_objects = "different kinds of bear"  # If multiple, add them like this: different kinds of bear, pepsi bottles, ... etc
number_of_prompts = 50  # Between 50 to 100 is recommended
description_of_prompt = "bear in different environments"  # A brief description of the desired images

# Generated Prompt
print(f'''
Important Objects that must be present in each prompt:
{important_objects}

Input:
Generate {number_of_prompts} realistic prompts related to {description_of_prompt} for image generation.

Instructions:
Each prompt depicts real-life behavior.
Each prompt must contain all the important objects.
The important objects must be at different levels of distance (from very close to the camera to very far).

Output:
Return a Python list containing these prompts as strings for later use in training a computer vision model.
prompts = [prompt1, prompt2, ...]
''')
```
By adjusting the parameters such as important_objects, number_of_prompts, and description_of_prompt, you can customize the prompts to suit the specific requirements of your computer vision project. The generated prompts will be crucial for obtaining a diverse and high-quality dataset for training your model.

## Step 2: Utilizing ChatGPT to Generate Realistic Image Descriptions

After creating the initial prompt template, the next step involves passing the generated prompt to ChatGPT for obtaining a Python list of strings containing the prompts. The output should be copied for further use.

<img src="https://cdn-images-1.medium.com/max/1000/1*wyvPYwUMWxu8kLVILftu6A.png" width=700>

Replace the example list with your custom prompts. In this instance, the focus is on detecting a specific animal, the bear.

```python
# Replace the following list with your own generated prompts
prompts = [
    "A grizzly bear fishing in a river with a distant mountain in the background and a fish jumping out of the water.",
    "A black bear climbing a tree to reach a beehive with a beehive in the foreground and a forest in the distance.",
    "A polar bear standing on an iceberg in the Arctic Ocean with other icebergs in the background and a seal in the water.",
    # ... (more prompts)
    "A teddy bear reading a book on a cozy couch by a fireplace with a rug and bookshelf in the background."
]

# The total number of generated prompts
len(prompts)
### Output is 50
```
Replace the prompts in the list with your specific requirements for the computer vision project. The diversity and specificity of these prompts play a crucial role in creating a high-quality and well-rounded dataset for training your computer vision model.

## Step 3: Increasing the number of images from limited prompts
In this step, we increase the number of prompts to 100 by doubling the existing prompts list. The prompts are then shuffled to add variety to the generated images.

```python
# Update the code to generate 100 images from 50 prompts (you can choose your own)
prompts = prompts * 2

# Shuffle the prompts
import random
random.shuffle(prompts)
```

This adjustment allows for a more extensive set of prompts, providing diversity for the subsequent image generation process.

## Step 4: Utilizing Realistic Vision Image Generation Model

Now, we employ a realistic vision image generation model to generate images based on the shuffled prompts. You can use your own model if needed.

Please note the important instructions mentioned:
1. Remove %%capture line if it throws any error.
2. Ensure GPU is enabled for this task.

```python
# Install the required library
%%capture
!pip install diffusers

# Import and load the realistic vision model
from diffusers import DiffusionPipeline
pipeline = DiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V5.0_noVAE")  # Loading the model

# Convert the model to GPU
pipeline = pipeline.to("cuda")

# Generate images and save them in the "generated_images" directory
generated_images = pipeline(prompts).images
```

## Step 5: Saving Generated Images
Finally, we save the generated images in a directory named /generated_images/ within the current working directory.

```python
# Import necessary libraries
import os

# Define the output directory name
output_directory = "generated_images"

# Save each image in the directory
for i, image in enumerate(generated_images):
    # Save the image with a unique filename
    image.save(os.path.join(output_directory, f'image_{i + 1}.png'))
```

This step ensures that the generated images are saved in the specified directory for later use in your computer vision project. Adjust the code as needed for your specific requirements.

Printing the sample of generated images

<img src="https://cdn-images-1.medium.com/max/1500/1*Tz35NjmqMqorsU_SE10vmQ.png">

## Step 6: Installing Autodistill and Creating Ontology
In this step, we install the required libraries, including Autodistill, and create an ontology to define what you want to detect in the generated images.

```python
# Install the necessary libraries using pip
!pip install -q autodistill autodistill-grounded-sam autodistill-yolov8 roboflow supervision==0.9.0

# Import the required module for defining the ontology
from autodistill.detection import CaptionOntology

# Define the ontology (Specify what you want to detect in your generated images)
ontology = CaptionOntology({
    "Bear": "Bear",
})
```

This step ensures that Autodistill and other related libraries are installed. Additionally, the ontology is created to provide a structured framework for auto-labeling the generated images based on the specified detection categories. Adjust the ontology categories according to your specific detection requirements.

## Step 7: Running Grounded SAM Model for Auto-Detection and Labeling
In this step, we utilize the Grounded SAM model to automatically detect and label the specified objects in the generated images. Ensure that Autodistill and Autodistill Grounded SAM have been successfully installed.

```python
# Import the Grounded SAM model from Autodistill
from autodistill_grounded_sam import GroundedSAM

# Create an instance of the Grounded SAM model using the defined ontology
base_model = GroundedSAM(ontology=ontology)

# Run the model to automatically detect and label objects in the generated images
dataset = base_model.label(
    input_folder='/content/generated_images',  # Specify the input folder containing the generated images
    extension=".png",  # Specify the file extension of the images
    output_folder='/content/output_folder'  # Specify the output folder for storing labeled data
)
```
This code snippet runs the Grounded SAM model on the generated images, automatically detecting and labeling the specified objects based on the ontology. Adjust the input and output folders as needed for your project. 
The labeled dataset will be stored in the specified output folder for further use in your computer vision project. After executing the provided code, the output will consist of images with labels corresponding to the selected object of interest. in txt format. 

Here are sample of the generated results:

<img src="https://cdn-images-1.medium.com/max/1500/1*obDvFQskmznn9u9uvdQJfA.png">
