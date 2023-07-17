import openai


openai.api_key = 'YOUR KEY'

prompt = '''Detect the category of the game based on the game's
description. categories are ['action', 'adventure', 'rpg','
strategy', 'simulation', 'sports and racing']. just return one
category. no explanation.
game's description: "Experience the thrill of high-speed racing
in this realistic driving simulation game. Choose from a
variety of vehicles, compete in exhilarating races, and prove
your skills on tracks around the world.'''

model = "text-davinci-002"
response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=500)
generated_text = response.choices[0].text
print(generated_text)