import openai

openai.api_key = 'YOUR KEY'

prompt = "Generate a dataset of game descriptions and their corresponding categories. The categories include action, adventure, RPG, strategy, simulation, and sports & racing. Please provide a game description and its category label. Each category should have 3 game descriptions."
model = "text-davinci-002"
response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=500)
generated_text = response.choices[0].text
print(generated_text)
