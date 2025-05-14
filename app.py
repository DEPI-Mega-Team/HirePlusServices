import streamlit as st
from streamlit_tags import st_tags

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if 'job_recommender_system_dir' not in st.session_state:
    st.session_state.job_recommender_system_dir = os.path.join(BASE_DIR, "JobRecommenderSystem")

if 'resume_analyzer_dir' not in st.session_state:
    st.session_state.resume_analyzer_dir = os.path.join(BASE_DIR, "ResumeAnalyzer")

if 'chatbot_dir' not in st.session_state:
    st.session_state.chatbot_dir = os.path.join(BASE_DIR, "ChatBot")

if 'current_dir' not in st.session_state:
    st.session_state.current_dir = BASE_DIR

if 'choice' not in st.session_state:
    st.session_state.choice = None

sys.path.insert(0, st.session_state.current_dir)

st.set_page_config(page_title="HirePlus Services", page_icon="🚀",)

# --- HEADER ---
st.markdown(
    """
    <h1 style='text-align: center; color: #4A90E2;'>🚀 HirePlus Data & AI Services</h1>
    <p style='text-align: center; font-size: 18px; color: gray;'>
        Select a tool below to view insights, enhance your job search, improve your resume, or practice interviews.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# --- APP NAVIGATION ---

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.choice = "Landing Page"

with col2:
    if st.button("📄 Resume Analyzer", use_container_width=True):
        st.session_state.choice = "Resume Analyzer"

with col3:
    if st.button("🤖 HireBot", use_container_width=True):
        st.session_state.choice = "Interview Chatbot"

with col4:
    if st.button("💼 Recommender", use_container_width=True):
        st.session_state.choice = "Job Recommender"

app_choice = st.session_state.choice
st.divider()


# --- LANDING PAGE ---
if app_choice in ["Landing Page", None]:
    st.title("Welcome to HirePlus Services!")
    st.image("https://via.placeholder.com/800x300.png?text=Welcome+to+HirePlus", use_container_width=True)
    st.markdown(
        """
        ### Why Choose HirePlus?
        - **📄 Resume Analyzer**: Quickly analyze your resume and streamline your job applications.
        - **🤖 Interview Chatbot**: Prepare for interviews with an AI-driven virtual assistant.
        - **💼 Job Recommender**: Get personalized job recommendations based on your skills and preferences.

        ### How It Works
        1. Select a tool from the navigation menu above.
        2. Follow the step-by-step instructions to use the selected tool.
        3. Leverage AI-powered insights to achieve your career aspirations!

        ### About Us
        At HirePlus, we are committed to empowering job seekers with innovative AI solutions. From enhancing your resume to preparing for interviews and finding the perfect job, our tools are designed to help you succeed in your career journey. Let’s work together to make your dream job a reality!
        """
    )
    st.markdown("---"           )
    st.subheader("\U0001F680 Explore the Modules")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("\U0001F4CB Resume Analyzer"):
            st.markdown("""
            - Quickly analyze your resume and streamline your job applications.
            - Extract key details and identify areas for improvement.
            """)
            if st.button("Go to Resume Analyzer", use_container_width= True):
                st.session_state.choice = "Resume Analyzer"

    with col2:
        with st.expander("\U0001F916 Interview Chatbot"):
            st.markdown("""
            - Practice interviews with an AI-driven virtual assistant.
            - Get personalized feedback on your responses.
            - Prepare for behavioral and technical interviews.
            """)
            if st.button("Go to Interview Chatbot", use_container_width= True):
                st.session_state.choice = "Interview Chatbot"

        with st.expander("\U0001F4E2 Job Recommender"):
            st.markdown("""
            - Get personalized job recommendations based on your skills and preferences.
            - Discover roles that align with your career goals.
            - Leverage AI to find your dream job.
            """)
            if st.button("Go to Job Recommender", use_container_width= True):
                st.session_state.choice = "Job Recommender"

# --- JOB RECOMMENDER APP ---
if app_choice == "Job Recommender":
    import torch
    torch.classes.__path__ = []
    
    sys.path[0] = st.session_state.job_recommender_system_dir

    from modules.recommender import job_recommender
    from modules.preprocessor import user_embedder, job_embedder
    from modules.utils import filter_recommendations
    from Models.models import user, job, weights
    from modules import consts as cs
    from modules.database import EmbeddingDB
    import pandas as pd
    import pickle
    import os
    from tqdm import tqdm
    import streamlit as st



    if 'users' not in st.session_state:
        st.session_state.users = None
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    
    if 'weights' not in st.session_state:
        st.session_state.weights = weights()

    if 'user_embed' not in st.session_state:
        print("Loading User Embedder ...")
        st.session_state.user_embed = user_embedder()
        print("Loaded User Embedder Successfully")

    if 'job_embed' not in st.session_state:
        print("Loading Job Embedder ...")
        st.session_state.job_embed = job_embedder()
        print("Loaded Job Embedder Successfully")

    if 'db' not in st.session_state:
        print("Loading Database ...")
        st.session_state.db = EmbeddingDB()
        print("Loaded Database Successfully")

    if 'recommender' not in st.session_state:
        print("Loading Recommender ...")
        st.session_state.recommender = job_recommender(st.session_state.db)
        print("Loaded Recommender Successfully")

    user_embed = st.session_state.user_embed
    job_embed = st.session_state.job_embed
    db = st.session_state.db
    recommender = st.session_state.recommender

    with st.spinner("Loading jobs..."):
        print("Loading jobs ...")
        jobs: pd.DataFrame = pickle.load(open(os.path.join(cs.workspace_dir, 'jobs.pkl'), 'rb'))[:1000]
        print("Jobs Loaded successfully")

        ids = [i for i in range(len(jobs))]

        job_objs = [None] * len(ids)
        for i, _id in enumerate(ids):
            jb = jobs[_id]
            job_objs[i] = job(
                job_id=i,
                title=jb['title'],
                content=jb['description'],
                work_type=jb['work_type']
            )

    with st.spinner("Loading job embeddings..."):
        missing_ids = db.get_missing_job_ids(ids)
        if missing_ids:
            for _id in tqdm(missing_ids):
                jb = job_objs[_id]
                embeddings = job_embed.embed(jb.model_copy())
                db.store_job_embeddings(jb.job_id, embeddings)

    print("Job Embeddings loaded successfully")

    if st.session_state.users is None:
        with st.sidebar and st.spinner("Loading users..."):
            user1 = {'title': 'Machine Learning Engineer',
                        'about': '''Machine Learning Engineer with a strong focus on building intelligent systems that bridge the gap between user needs and actionable insights. Currently, I am working on developing a sophisticated recommender system designed to match user profiles with job descriptions, leveraging advanced natural language processing (NLP) and collaborative filtering techniques. My role involves end-to-end ownership of the project, from data preprocessing and feature engineering to model training, evaluation, and deployment.

            With a solid foundation in Computer Science, I bring 4 years of experience in designing and implementing machine learning solutions that drive real-world impact. My expertise includes working with large-scale datasets, optimizing recommendation algorithms, and deploying models into production environments using cloud platforms like AWS and GCP. I am proficient in Python, TensorFlow, PyTorch, and Scikit-learn, and have hands-on experience with tools like Spark and Docker for scalable data processing and deployment.

            In my current project, I am focused on improving the accuracy and personalization of job recommendations by incorporating user behavior data, contextual information, and advanced NLP techniques to better understand both user profiles and job descriptions. This involves experimenting with deep learning architectures, such as BERT-based models, and continuously iterating to enhance system performance. For example, I recently implemented a hybrid recommendation approach that combines content-based filtering with matrix factorization, resulting in a 15% improvement in recommendation relevance based on user feedback.

            I am passionate about creating ethical, transparent, and user-centric AI systems that deliver meaningful value. Outside of work, I enjoy staying updated with the latest advancements in AI/ML, contributing to open-source projects like Hugging Face Transformers, and participating in technical communities such as Kaggle and Meetup groups. I am always open to connecting with professionals who share a passion for innovation and problem-solving. Let’s connect and explore how we can collaborate to build the future of intelligent systems!''',
                        'preferred_work_types': ['FULL_TIME', 'INTERNSHIP'],
                        'experience_level': 'Entry level',
                        'expected_salary': None,
                        'skills': ['python', 'scikitlearn', 'tensorflow', 'pandas']
                        }
            user2 = {
                'title': 'Graphics Designer',
                'about': '''Creative Graphics Designer with 5 years of experience in crafting visually compelling designs for digital and print media. Skilled in Adobe Creative Suite, including Photoshop, Illustrator, and InDesign, with a strong focus on branding, marketing materials, and user interface design. Passionate about translating client visions into impactful visual stories that resonate with target audiences.

            In my current role, I have successfully led design projects for various industries, delivering high-quality work under tight deadlines. My expertise includes logo design, social media graphics, and website mockups, ensuring consistency in brand identity across all platforms. I am also proficient in motion graphics and video editing, using tools like After Effects and Premiere Pro to create engaging multimedia content.

            I thrive in collaborative environments, working closely with marketing teams, developers, and clients to achieve project goals. My attention to detail and commitment to excellence have earned me recognition for delivering designs that exceed expectations. Outside of work, I enjoy exploring new design trends, participating in design challenges, and mentoring aspiring designers.''',
                'preferred_work_types': ['FULL_TIME', 'PART_TIME'],
                'experience_level': 'Mid level',
                'expected_salary': None,
                'skills': ['photoshop', 'illustrator', 'indesign', 'aftereffects']
            }

            user3 = {
                'title': 'Digital Marketer',
                'about': '''Results-driven Digital Marketer with 6 years of experience in developing and executing data-driven marketing strategies to boost brand awareness and drive customer engagement. Proficient in SEO, SEM, social media marketing, email campaigns, and analytics tools like Google Analytics and HubSpot. Adept at creating targeted campaigns that deliver measurable ROI.

            In my current role, I have successfully managed multi-channel marketing campaigns, increasing website traffic by 40% and improving conversion rates by 25%. My expertise includes content marketing, pay-per-click advertising, and influencer collaborations, ensuring alignment with business objectives. I am also skilled in A/B testing and performance analysis to optimize campaign effectiveness.

            I am passionate about staying ahead of digital marketing trends and leveraging innovative techniques to connect with audiences. Outside of work, I enjoy attending marketing conferences, contributing to industry blogs, and networking with professionals in the field.''',
                'preferred_work_types': ['PART_TIME', 'CONTRACT'],
                'experience_level': 'Mid level',
                'expected_salary': None,
                'skills': ['seo', 'sem', 'googleanalytics', 'hubspot']
            }
            users = [user1, user2, user3]
            for i, usr_data in enumerate(users):
                usr = user(
                    user_id= i + 1,
                    title=usr_data['title'],
                    about=usr_data['about'],
                    preferred_work_types=usr_data['preferred_work_types'],
                    experience_level=usr_data['experience_level'],
                    expected_salary=usr_data['expected_salary'],
                    skills=usr_data['skills']
                )
                
                users[i] = usr

                print(f"Embedding user {usr.user_id} ...")
                embeddings = user_embed.embed(usr.model_copy())
                print(f"Storing user {usr.user_id} embeddings ...")
                db.store_user_embeddings(usr.user_id, embeddings)
                print(f"User {usr.user_id} embeddings stored successfully")
            
            st.session_state.users = users


    st.title("Job Recommender System")
    st.write("This is a job recommender system that recommends jobs based on user profiles.")
    st.divider()
    users = st.session_state.users
    weight = st.session_state.weights

    with st.sidebar:
        select_user = st.selectbox("Select a user", options=[f"User {usr.user_id}: {usr.title}" for usr in users] + ["Custom User"])
        
        if select_user == "Custom User":
            title = st.text_input("Title")
            about = st.text_area("About")
            preferred_work_types = st.multiselect("Preferred Work Types", options=['CONTRACT', 'FULL_TIME', 'INTERNSHIP', 'PART_TIME', 'TEMPORARY', 'VOLUNTEER'])
            
            usr = user(
                user_id=0,
                title= title,
                about= about,
                preferred_work_types= preferred_work_types,
                experience_level= None,
                expected_salary=None,
                skills= None
            )
        else:
            usr = users[int(select_user[5]) - 1]
            st.header("User Information")
            st.markdown(f"**Title:**")
            st.text(f"{usr.title}")
            st.markdown(f"**About:**")
            st.text(f"{usr.about[:200]} ...")
            st.markdown(f"**Preferred Work Types:** {', '.join(usr.preferred_work_types)}")
        
        button_recommend = st.button("Recommend Jobs", use_container_width=True)
        
        st.markdown("---")
        st.header("Specify Recommender Weights")
        
        title = st.slider("Title", min_value=0.0, max_value=1.0, step=0.1, value=weight.title)
        content = st.slider("About", min_value=0.0, max_value=1.0, step=0.1, value=weight.content)
        work_type = st.slider("Work Type", min_value=0.0, max_value=1.0, step=0.1, value=weight.work_type)
        st.markdown(f"**Summation of weights:** {title + content + work_type}")
        
        set_weights = st.button("Set Weights", use_container_width=True)
        if set_weights:
            weight.title = title
            weight.content = content
            weight.work_type = work_type
            
            if weight.title + weight.content + weight.work_type != 1.0:
                st.warning("The sum of weights should equal 1.0")
            else:
                st.session_state.weights = weight


        if button_recommend:
            if select_user == "Custom User":
                print(f"Embedding user {usr.user_id} ...")
                embeddings = user_embed.embed(usr)
                print(f"Storing user {usr.user_id} embeddings ...")
                db.store_user_embeddings(usr.user_id, embeddings)
                print(f"User {usr.user_id} embeddings stored successfully")

            recommendations = recommender.user_job_recommend(
                user_id= usr.user_id,
                jobs_ids= [jb.job_id for jb in job_objs],
                recommender_weights= st.session_state.weights,
            )
            st.session_state.recommendations = filter_recommendations(recommendations, max_recommendations=10)


    if st.session_state.recommendations:
        st.header("Recommended Jobs")
        for job_id in st.session_state.recommendations:
            job = job_objs[job_id]
            with st.container():
                st.markdown(f"### {job.title}")
                st.markdown(f"**Work Type:** {job.work_type}")
                st.markdown(f"**Description:** {job.content[:200]}...")
                if st.button(f"View More", key=f"view_more_{job_id}"):
                    st.markdown(f"**Full Description:** {job.content}")
                st.markdown("---")

# --- RESUME ANALYZER APP ---
elif app_choice == "Resume Analyzer":
    import torch
    import io

    torch.classes.__path__ = []
    
    sys.path[0] =  st.session_state.resume_analyzer_dir

    if 'parser' not in st.session_state:
        from resume_analyzer import ResumeParser
        st.session_state.parser = ResumeParser()

    if 'resume_data' not in st.session_state:
        st.session_state.resume_data = None

    parser = st.session_state.parser
    resume = st.file_uploader("Upload your resume", type=["pdf", "docx"], accept_multiple_files= False)
    if not resume:
        st.stop()

    parse_button = st.button("Parse Resume", use_container_width= True)
    if parse_button:
        st.session_state.resume_data = None
        resume_content = resume.read()
        resume_io = io.BytesIO(resume_content)
        resume_io.name = resume.name
        details = parser.parse(resume_io)
        st.session_state.resume_data = details

    if st.session_state.resume_data:
        for key, value in st.session_state.resume_data.items():
            if isinstance(value, list):
                keywords = st_tags(label=key, value= value)
                continue
            if isinstance(value, float) or isinstance(value, int):
                st.number_input(key, value= value)
                continue
            else:
                st.text_input(key, value)

# --- INTERVIEW CHATBOT APP ---
elif app_choice == "Interview Chatbot":
    # Move current working directory to ChatBot Directory
    working_dir = st.session_state.chatbot_dir
    
    sys.path[0] = st.session_state.chatbot_dir
    
    from langchain_core.messages import AIMessage, HumanMessage
    from app.chains import interview_chains, instruction_chain

    
    st.title("HireBot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "data" not in st.session_state:
        st.session_state.data = {
            'candidate_name': None, 
            'job_title': None, 
            'interview_type': None, 
            'number_of_questions': None
        }

    if "instructions" not in st.session_state:
        st.session_state.instructions = ""

    roles = {HumanMessage: "user", AIMessage: "assistant"}

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        role = type(message)
        with st.chat_message(roles[role]):
            st.markdown(message.content)


    def chat(prompt= None):
        # Append user message to chat history
        if not prompt:
            history = [HumanMessage(content="Hello")]
        else:
            st.session_state.messages.append(HumanMessage(content=prompt))
            history = st.session_state.messages
        
        chain_input = {
            "candidate_name": st.session_state.data['candidate_name'],
            "job_title": st.session_state.data['job_title'],
            "number_of_questions": st.session_state.data['number_of_questions'],
            "skills": st.session_state.data['skills'],
            "instructions": st.session_state.instructions,
            "chat_history": history,
        }
        
        chain = interview_chains[st.session_state.data['interview_type']]
        # Generate response
        response = chain.invoke(chain_input)
        
        # Append AI response to chat history
        st.session_state.messages.append(AIMessage(content=response))
        
        # Rerun the app to display the new messages
        st.rerun()

    # Sidebar for user input
    with st.sidebar: # Initialize Side Bar Items
        st.title("Side Panel")
        
        # Collect user input
        name = st.text_input("Candidate Name")
        job_title = st.text_input("Job Title")
        interview_type = st.selectbox("Interview Type", ["behavioral", "technical"])
        number_of_questions = st.number_input("Number of Questions", min_value=1, max_value=10, value=5)
        skills = st_tags(label="Enter Your Skills")
        
        # Store user input in session state
        st.session_state.data['candidate_name'] = name
        st.session_state.data['job_title'] = job_title
        st.session_state.data['interview_type'] = interview_type
        st.session_state.data['number_of_questions'] = number_of_questions
        st.session_state.data['skills'] = skills
        
        # Button to start a new interview
        generate = st.button("Start New Interview")


    if generate:
        # Empty chat history 
        st.session_state.messages =  []
        
        # Generate instructions
        st.session_state.instructions= instruction_chain.invoke({
                                    "candidate_name": st.session_state.data['candidate_name'],
                                    "job_title": st.session_state.data['job_title'],
                                    "number_of_questions": st.session_state.data['number_of_questions'],
                                    "skills": st.session_state.data['skills'],
                                    'interview_type': st.session_state.data['interview_type'],
                                    })
        
        # Start New Chat
        chat()

    prompt = st.chat_input("Chat with HireBot")

    if st.session_state.instructions and prompt:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Proceed in chat
        chat(prompt)
