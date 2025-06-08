import os
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
import argparse

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
tmodel = 'gemini-2.5-flash-preview-04-17'
imodel = 'imagen-3.0-generate-002'
output_path = '/home/jrm/dev/oai/output'

def upload(file_path: str) -> str:
    # check if file exists
    if not os.path.exists(file_path):
        raise Exception("File does not exist")

    return client.files.upload(file=file_path, config={
        'display_name': os.path.basename(file_path)
    })    

def describe(file, instructions="",temperature=1.2, top_k=100):
    lens = ''
    if instructions:
        lens = 'Lens: ' + instructions

    response = client.models.generate_content(
        model=tmodel,
        contents=[
            file,
            'describe image for use by a generative image ai to recreate it.',
            'if you are given a "Lens" use it to create the description. adhere to lens as strictly as possible',
            'give only the description, no commentary.',
            lens,
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            top_k=top_k
        )

    )
    return response.text

def style(file, instructions='',temperature=1.2, top_k=100):
    response = client.models.generate_content(
            model=tmodel,
            contents=[
                file,
                'describe the artistic style of the image.',
                'do not include subject, just the mechanical aspects.',
                'no commentary, just result',
                instructions
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                top_k=top_k
            )
    )
    
    return response.text

def restyle(prompt, style, temperature=1.2, top_k=100):
    response = client.models.generate_content(
            model=tmodel,
            contents=[
                'rework <input> with regard to <style>.',
                'attempt to keep subject the same as possible',
                'no commentary, only result',
                '<input>' + prompt + '</input>',
                '<style>' + style + '</style>'
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                top_k=top_k
            )
    )
    return response.text


def blend(a, b, strength, temperature=1.2, top_k=100):
    response = client.models.generate_content(
            model=tmodel,
            contents=[
                'blend <a> and <b> using <strength> where <strength> is weight of <a> over <b>. <strength>50</strength> would be an equal blend.',
                'no commentary, just the result',
                '<a>' + a + '</a>',
                '<b>' + b + '</b>',
                '<strength>' + strength + '</strength>'
            ]
    )
    return response.text


def mutate(prompt, lens, temperature=1.2, top_k=100):
    response = client.models.generate_content(
            model=tmodel,
            contents=[
                'you will adopt the frame within <f></f> completely.',
                'with it, recontextualize the input <i></i>.',
                'no commentary, just the result.',
                '<f>' + lens + '</f>',
                '<i>' + prompt + '</i>'
            ]
    )

    return response.text

def imagine(subject, prompt, temperature=1.2, top_k=100):
    response = client.models.generate_content(
            model=tmodel,
            contents=[
                subject,
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                top_k=top_k,
                max_output_tokens=256
            )
    )
    return response
    
def generate(prompt):
    response = client.models.generate_images(
        model=imodel,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images= 1,
            include_rai_reason= True,
        )
    )
    return response.generated_images[0].image

def get_next_filename(path, prefix, ext='png'):
    # look in path for files like <prefix><index>.<ext>
    # find the highest index
    # return <prefix><index+1>.<ext>
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get all files in the directory
    files = os.listdir(path)
    
    # Filter files matching the pattern
    pattern = f"{prefix}(\\d+)\\.{ext}"
    indices = []
    
    for file in files:
        match = re.search(pattern, file)
        if match:
            indices.append(int(match.group(1)))
    
    # Find the next index
    next_index = 1
    if indices:
        next_index = max(indices) + 1
    
    return os.path.join(path, f"{prefix}{next_index}.{ext}")

def save(image, prefix):
    filename = get_next_filename(output_path, prefix)
    image.save(filename)
    return filename

class Files:
    def __init__(self):
        self.updatefiles()

    def get(self, name):
        for file in self.files:
            if file.display_name == name:
                return GenFile(file)
        return None

    def list(self):
        return [x for x in self.files]

    def upload(self, filename):
        f = GenFile(upload(filename))
        self.updatefiles()

        return f

    def updatefiles(self):
        self.files = client.files.list()

class GenFile:
    def __init__(self, file, temperature=1.2, top_k=100):
        self.file = file
        self.temperature = temperature
        self.top_k = top_k

    def describe(self, instructions=''):
        text = describe(self.file, instructions, self.temperature, self.top_k)
        return GenText(text, self.temperature, self.top_k)

    def style(self, file):
        text = style(self.file, self.temperature, self.top_k)
        return GenText(text, self.temperature, self.top_k)

class GenImg:
    def __init__(self, image):
        self.image = image

    def save(self, prefix):
        save(self.image, prefix)

class GenText:
    def __init__(self, text='', temperature=1.2, top_k=100):
        self.text = text
        self.temperature = temperature
        self.top_k = top_k

    def __repr__(self):
        return self.text

    @classmethod
    def imagine(cls, subject, prompt):
        text = imagine(subject, prompt)
        return GenText(text)

    def restyle(self, prompt, style):
        self.text = restyle(prompt, style, self.temperature, self.top_k)
        return GenText(self.text, self.temperature, self.top_k)

    def blend(self, a, b, strength):
        self.text = blend(a, b, strength, self.temperature, self.top_k)
        return GenText(self.text, self.temperature, self.top_k)

    def mutate(self, prompt, lens):
        self.text = mutate(prompt, lens, self.temperature, self.top_k)
        return GenText(self.text, self.temperature, self.top_k)

    def generate(self):
        image = generate(self.text)
        return GenImg(image)

if __name__ == '__main__':
    files = Files()

    t = GenText.imagine('a jealous hammer teaches velvet frogs about time',
                        'create an image description from this scene')
    print(t)
    t.generate().save('x')
