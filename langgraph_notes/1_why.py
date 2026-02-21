# Why we need Agentic AI ?

"""
? What is Generative Al ?
Generative Al refers to a class of artificial intelligence models
that can create new content - such as text, images, audio,
code, or video - that resembles human-created data.

-> LLMs based apps like Chat GPTTTS models like ElevenLabs
-> Diffusion models for images
-> Code generating LLMs like

? Generative AI vs Traditional AI ?
-> Traditional AI is about finding patterns in data and giving predictions.
-> Generative AI is about learning the distribution of data so that it can generate a new sample from it.

? Goal : To Hire a backend engineer!
1) Craeting a JD
2) Posting the JD to a job platform
3) Shortlisting
4) Interviewing
5) Rolling an offer letter
6) Onboarding

? How to execute  this complete task with the help of GenAI ?
1) LLM Based Chatbot
    -> JD Drafting (Generate a JD for the <role>)
    -> Job Posting (Where I can post this job role)
       Response like this: Post the role on LinkedIn, Naukari, Indeed manually.
    -> Shortlisitng (Ask LLM for what skill need to check for shortlisting candidates (review resumes manually))
    -> Scheduling (Ask LLM to Draft an Email for scheduling the interview)
    -> Interviewing (Ask LLM to give you some question for interview)
    -> Drafting OfferLetter (Ask LLM to draft an offer letter for selected candidates)

    ? Problems in Above Approach
    ! 1) Reactive (Chatbot not able to take decision on its own (Human Intervention needed))
    ! 2) No Memory
    ! 3) Generic Advice (Don't give company specific results)
    ! 4) Can't take actions on its own (like it can provide email content but can't send email)
    ! 5) Can't adopt (means if flow of action is not going well it will have to choose alternatives path to solve the issue.)

2) RAG Based Chatbot (This will solve 3rd problem: Generic Advice)
    -> Which Kind of Data need to provide (in the context of this goal) ?
        -> JD Templates 
            -> Examples of High Performing JD's
            -> Past JD's used by company
        -> Hiring Strategies
            -> Best platform for hiring
            -> Best Practice for hirings
            -> Internal Salary band
            -> Internal checklist to hire for various profiles
            -> Interview question banks
        -> Onboarding Checklist
            -> Offer letter template
            -> Welcome Email template
            -> Employee policies
            -> etc.

3) Tools Augmented Chatbot (This will solve 4th problem.)
    -> calender API
    -> linkedin API
    -> Resume Parser 
    -> Mail API
    -> HRM Access Tools

4) Agentic AI Chatbot (This will solve all the remaining 3 problems.)
    -> Autonomous decision-making: Can choose the best course of action without human intervention.
    -> Dynamic goal adaptation: Adjusts strategies if initial plans fail or new information arises.
    -> Multi-step task execution: Handles complex workflows end-to-end (e.g., from JD creation to onboarding).
    -> Memory and context retention: Remembers previous interactions and uses them to inform future actions.
    -> Integration with external tools: Directly interacts with APIs (email, calendar, HRM, job platforms) to perform actions.
    -> Personalized recommendations: Tailors advice and actions based on company-specific data and past outcomes.
    -> Continuous learning: Improves its processes over time by learning from successes and failures.
    -> Proactive problem-solving: Identifies potential issues and suggests or implements solutions before they escalate.
    -> Transparent reasoning: Explains its decisions and actions to users for trust and accountability.

? Conclusion
-> Generative AI is about creating the content, Agentic AI is about solveing a goal.
-> Generative AI is reactive, Agentic AI is proactive (autonomous)
-> Generative AI is a building block of Agentic AI.
-> Generative AI is a capability whereas Agentic AI is a behaviour.
"""