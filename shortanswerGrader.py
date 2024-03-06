from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
import os

os.environ["OPENAI_API_KEY"] = "your key"

template = '''
{text}

Grade these questions and answers and give a final score out of 10
give the score as output only nothing else in the format score'''

prompt = PromptTemplate(
    input_variables=['text'],
    template=template
)

questions_answers = '''
Q:"What are the main causes of climate change?"
Answer:Climate change is primarily caused by human activities such as burning fossil fuels, deforestation, industrial processes, and agricultural practices. These activities release greenhouse gases such as carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O) into the atmosphere, trapping heat and leading to global warming. Additionally, other factors such as volcanic eruptions, solar radiation, and natural variations in Earth's orbit and axial tilt also contribute to climate change, but their impact is relatively minor compared to human-induced factors.
'''

llm = OpenAI(temperature=0.6)
chain1 = LLMChain(llm=llm, prompt=prompt)

output = chain1.invoke(questions_answers)
score=output['text']
print(score)
