import os  # for loading api from .env file
import textwrap  # for formatting the output
from time import monotonic  # Times the run time of the chain

import openai
import tiktoken  # for getting the encoding of the model
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI  # for generating the summary
from langchain.docstore.document import Document  # for storing the text
from langchain.prompts import PromptTemplate  # Template for the prompt
from langchain.text_splitter import (
    CharacterTextSplitter,  # for splitting the text into chunks
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Summarization functin
def summarize_test():
    model_name = "gpt-3.5-turbo"
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        model_name=model_name,
    )
    text = "There has been a remarkable surge in the use of algorithms and artificial intelligence to address a wide range of problems and challenges. While their adoption, particularly with the rise of AI, is reshaping nearly every industry sector, discipline, and area of research, such innovations often expose unexpected consequences that involve new norms, new expectations, and new rules and laws.\nTo facilitate deeper understanding, the Social and Ethical Responsibilities of Computing (SERC), a cross-cutting initiative in the MIT Schwarzman College of Computing, recently brought together social scientists and humanists with computer scientists, engineers, and other computing faculty for an exploration of the ways in which the broad applicability of algorithms and AI has presented both opportunities and challenges in many aspects of society.\n\u201cThe very nature of our reality is changing. AI has the ability to do things that until recently were solely the realm of human intelligence \u2014 things that can challenge our understanding of what it means to be human,\u201d remarked Daniel Huttenlocher, dean of the MIT Schwarzman College of Computing, in his opening address at the inaugural SERC Symposium. \u201cThis poses philosophical, conceptual, and practical questions on a scale not experienced since the start of the Enlightenment. In the face of such profound change, we need new conceptual maps for navigating the change.\u201d\nThe symposium offered a glimpse into the vision and activities of SERC in both research and education. \u201cWe believe our responsibility with SERC is to educate and equip our students and enable our faculty to contribute to responsible technology development and deployment,\u201d said Georgia Perakis, the William F. Pounds Professor of Management in the MIT Sloan School of Management, co-associate dean of SERC, and the lead organizer of the symposium. \u201cWe\u2019re drawing from the many strengths and diversity of disciplines across MIT and beyond and bringing them together to gain multiple viewpoints.\u201d\nThrough a succession of panels and sessions, the symposium delved into a variety of topics related to the societal and ethical dimensions of computing. In addition, 37 undergraduate and graduate students from a range of majors, including urban studies and planning, political science, mathematics, biology, electrical engineering and computer science, and brain and cognitive sciences, participated in a poster session to exhibit their research in this space, covering such topics as quantum ethics, AI collusion in storage markets, computing waste, and empowering users on social platforms for better content credibility.\nShowcasing a diversity of work\nIn three sessions devoted to themes of beneficent and fair computing, equitable and personalized health, and algorithms and humans, the SERC Symposium showcased work by 12 faculty members across these domains.\nOne such project from a multidisciplinary team of archaeologists, architects, digital artists, and computational social scientists aimed to preserve endangered heritage sites in Afghanistan with digital twins. The project team produced highly detailed interrogable 3D models of the heritage sites, in addition to extended reality and virtual reality experiences, as learning resources for audiences that cannot access these sites.\nIn a project for the United Network for Organ Sharing, researchers showed how they used applied analytics to optimize various facets of an organ allocation system in the United States that is currently undergoing a major overhaul in order to make it more efficient, equitable, and inclusive for different racial, age, and gender groups, among others.\nAnother talk discussed an area that has not yet received adequate public attention: the broader implications for equity that biased sensor data holds for the next generation of models in computing and health care.\nA talk on bias in algorithms considered both human bias and algorithmic bias, and the potential for improving results by taking into account differences in the nature of the two kinds of bias.\nOther highlighted research included the interaction between online platforms and human psychology; a study on whether decision-makers make systemic prediction mistakes on the available information; and an illustration of how advanced analytics and computation can be leveraged to inform supply chain management, operations, and regulatory work in the food and pharmaceutical industries.\nImproving the algorithms of tomorrow\n\u201cAlgorithms are, without question, impacting every aspect of our lives,\u201d said Asu Ozdaglar, deputy dean of academics for the MIT Schwarzman College of Computing and head of the Department of Electrical Engineering and Computer Science, in kicking off a panel she moderated on the implications of data and algorithms.\n\u201cWhether it\u2019s in the context of social media, online commerce, automated tasks, and now a much wider range of creative interactions with the advent of generative AI tools and large language models, there\u2019s little doubt that much more is to come,\u201d Ozdaglar said. \u201cWhile the promise is evident to all of us, there\u2019s a lot to be concerned as well. This is very much time for imaginative thinking and careful deliberation to improve the algorithms of tomorrow.\u201d\nTurning to the panel, Ozdaglar asked experts from computing, social science, and data science for insights on how to understand what is to come and shape it to enrich outcomes for the majority of humanity.\nSarah Williams, associate professor of technology and urban planning at MIT, emphasized the critical importance of comprehending the process of how datasets are assembled, as data are the foundation for all models. She also stressed the need for research to address the potential implication of biases in algorithms that often find their way in through their creators and the data used in their development. \u201cIt\u2019s up to us to think about our own ethical solutions to these problems,\u201d she said. \u201cJust as it\u2019s important to progress with the technology, we need to start the field of looking at these questions of what biases are in the algorithms? What biases are in the data, or in that data\u2019s journey?\u201d\nShifting focus to generative models and whether the development and use of these technologies should be regulated, the panelists \u2014 which also included MIT\u2019s Srini Devadas, professor of electrical engineering and computer science, John Horton, professor of information technology, and Simon Johnson, professor of entrepreneurship \u2014 all concurred that regulating open-source algorithms, which are publicly accessible, would be difficult given that regulators are still catching up and struggling to even set guardrails for technology that is now 20 years old.\nReturning to the question of how to effectively regulate the use of these technologies, Johnson proposed a progressive corporate tax system as a potential solution. He recommends basing companies' tax payments on their profits, especially for large corporations whose massive earnings go largely untaxed due to offshore banking. By doing so, Johnson said that this approach can serve as a regulatory mechanism that discourages companies from trying to \u201cown the entire world\u201d by imposing disincentives.\nThe role of ethics in computing education\nAs computing continues to advance with no signs of slowing down, it is critical to educate students to be intentional in the social impact of the technologies they will be developing and deploying into the world. But can one actually be taught such things? If so, how?\nCaspar Hare, professor of philosophy at MIT and co-associate dean of SERC, posed this looming question to faculty on a panel he moderated on the role of ethics in computing education. All experienced in teaching ethics and thinking about the social implications of computing, each panelist shared their perspective and approach.\nA strong advocate for the importance of learning from history, Eden Medina, associate professor of science, technology, and society at MIT, said that \u201coften the way we frame computing is that everything is new. One of the things that I do in my teaching is look at how people have confronted these issues in the past and try to draw from them as a way to think about possible ways forward.\u201d Medina regularly uses case studies in her classes and referred to a paper written by Yale University science historian Joanna Radin on the Pima Indian Diabetes Dataset that raised ethical issues on the history of that particular collection of data that many don\u2019t consider as an example of how decisions around technology and data can grow out of very specific contexts.\nMilo Phillips-Brown, associate professor of philosophy at Oxford University, talked about the Ethical Computing Protocol that he co-created while he was a SERC postdoc at MIT. The protocol, a four-step approach to building technology responsibly, is designed to train computer science students to think in a better and more accurate way about the social implications of technology by breaking the process down into more manageable steps. \u201cThe basic approach that we take very much draws on the fields of value-sensitive design, responsible research and innovation, participatory design as guiding insights, and then is also fundamentally interdisciplinary,\u201d he said.\nFields such as biomedicine and law have an ethics ecosystem that distributes the function of ethical reasoning in these areas. Oversight and regulation are provided to guide front-line stakeholders and decision-makers when issues arise, as are training programs and access to interdisciplinary expertise that they can draw from. \u201cIn this space, we have none of that,\u201d said John Basl, associate professor of philosophy at Northeastern University. \u201cFor current generations of computer scientists and other decision-makers, we\u2019re actually making them do the ethical reasoning on their own.\u201d Basl commented further that teaching core ethical reasoning skills across the curriculum, not just in philosophy classes, is essential, and that the goal shouldn\u2019t be for every computer scientist be a professional ethicist, but for them to know enough of the landscape to be able to ask the right questions and seek out the relevant expertise and resources that exists.\nAfter the final session, interdisciplinary groups of faculty, students, and researchers engaged in animated discussions related to the issues covered throughout the day during a reception that marked the conclusion of the symposium.\n"

    texts = text_splitter.split_text(text)  # splits the text into chunks
    docs = [Document(page_content=t) for t in texts]  # Converts each part into a Document object

    # Loads the model
    llm = ChatOpenAI(temperature=0, openai_api_key=openai.api_key, model_name=model_name)

    # Defines prompt template
    prompt_template = """Write a consise summary of the following text:{text}"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Function that counts the number of tokens in a string
    def num_tokens_from_string(string, encoding_name):
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    # Calculates the number of tokens in the text
    num_tokens = num_tokens_from_string(text, model_name)

    # Model max tokens
    model_max_tokens = 4097
    verbose = False  # If set to True, prints entire un-summarized text

    # Loads appropriate chain based on the number of tokens
    # Stuff or Map Reduce is chosen
    if num_tokens < model_max_tokens:
        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=prompt,
            verbose=verbose,
        )
    else:
        chain = load_summarize_chain(
            llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt, verbose=verbose
        )

    # Records the time it takes to run the chain
    # start_time = monotonic() # Optional run time of program
    summary = chain.run(docs)

    # Prints the results
    # print(f"Chain type: {chain.__class__.__name__}") # Optional chain type (stuff or map_reduce)
    # print(f"Run time: {monotonic() - start_time}") # Optional run time of program
    print(f"Summary: {textwrap.fill(summary, width=100)}")


# Runs the summarize_test function (prints in terminal)
if __name__ == "__main__":
    summarize_test()
