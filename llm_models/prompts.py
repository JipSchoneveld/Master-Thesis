"""
File with all prompt elements and prompt construction functions 
"""

from itertools import product
import pandas as pd 
from textwrap import dedent
import re 

class PromptItem:
    """
    Class which defines prompt items with attributes:
    - text: text of prompt item 
    - desc: short identifer of the prompt item 
    """

    def __init__(self, text, desc: str):
        self.text = text
        self.desc = desc


TASK = [
    PromptItem("Given this clinical note, determine the progression status of {symptom} (defined as: {description}) within the treatment trajectory.", "T:i"),
    PromptItem("What does this clinical note tell us about the progression status of {symptom} (defined as: {description}) within the treatment trajectory?", "T:q")
]

# prompt text is saved as dictionary with a text for 'mention' and 'pregression' prompting 
TASK_2STEP = [
    PromptItem(
        {
            'mention': "Given this clinical note, determine if {symptom} (defined as: {description}) is relevant for or experienced by the patient within the treatment trajectory.",
            'progression': "Given this clinical note, determine the progression status of {symptom} (defined as: {description}) within the treatment trajectory."
        }, 
        "T:i"
    ),
    PromptItem(
        {
            'mention': "Does this clinical note tell us that {symptom} (defined as: {description}) is relevant for or experienced by the patient within the treatment trajectory?",
            'progression': "What does this clinical note tell us about the progression status of {symptom} (defined as: {description}) within the treatment trajectory?"
        }, 
        "T:q"
    )
]
  
ROLE = [
    PromptItem("You are a clinician. ", "R:c"), 
    PromptItem("", "R:-")
]

OVERALL_GOAL = [
    PromptItem("The goal is to determine how symptoms progress during treatment trajectories. ", "OG:d"),
    PromptItem("The goal is to extract treatment outcomes from clinical notes. ", "OG:i"),
    PromptItem("", "OG:-")
]

PATIENT_INFO = [
    PromptItem("\nThe clinical note is from a patient who has received a depression diagnosis.", "PI:d"),
    PromptItem("\nThe clinical note is from a patient who has symptoms of depression.", "PI:s"),
    PromptItem("", "PI:-")
]

SYMPTOMS = {
    0: ("depressed mood", "depressed mood, gloominess, despondence, or hopelessness; including non-verbal signs of this"),
    1: ("self depreciation and feelings of guilt", "low self-esteem, self-reproach, lack of self-respect, feelings of guilt, or feeling they deserve punishment"),
    2: ("suicidal tendency", "feeling that life is not worth living, desire to die, consideration of, plans for, or recent attempts at suicide"),
    3: ("insomnia", "issues with falling asleep, staying asleep, or waking up prematurely"), 
    4: ("interests and activities", "anhedonia, lack of interests or motivation, reduced activity level, inability or reduced ability to perform work or activities, needing to force themselves to perform work or activities"), 
    5: ("retardation", "slowness of movement, speech, facial expressions, or gestures"),
    6: ("agitation", "agitation, physical restlessness, difficulty sitting still"),
    7: ("anxiety (psychological)", "anxiety, being scared, feeling unsafe or threatened, panic, nervousness, worry, irritability"),
    8: ("somatic", "physical manifestations and symptoms linked to psychological issues: gastro-intestinal issues, reduced appetite or food intake, lack of taste, constipation, cramps, excessive sweating, trembling, hyperventilation, dry mouth, (muscle)aches, physical tiredness, lack of energy, exhaustion, heavy feeling in the limbs"),
    9: ("hypochondria", "excessive concern about physical health, excessive preoccupation with physical sensations, excessive preoccupation with the physical symptoms of the depression, thinking they have a serious illness, hypochondriac delusions"),
    10: ("lack of insight", "denial of the presence or severity of depression, attributing the depression to external and unrelated factors"), 
    11: ("weight", "actual loss of weight") 
}

LABELS = """\
    A. Established/unchanged: the patient is experiencing this symptom, but there is no indication that the occurrence of the symptom is a recent development in the trajectory or that there is any change in severity, or it is explicitly mentioned that the severity is unchanged, or the symptoms description clearly serves to provide an overview of all relevant symptoms.
    B. Worsened: the severity the patient experiences of this symptom has gotten worse. This can also be indicated by a description that the patient has started experiencing the symptom recently.
    C. Improved: the severity the patient experiences of this symptom has gotten better/less. This can also be indicated by a description that the patient has recently experienced symptom opposites. 
    D. No mention: neither the symptom nor an opposite is mentioned in the note, or it is explicitly mentioned that the symptom is not relevant for, or experienced by the patient."""

LABELS_PROGRESSION = """\
    A. Established/unchanged: the patient is experiencing this symptom, but there is no indication that the occurrence of the symptom is a recent development in the trajectory or that there is any change in severity, or it is explicitly mentioned that the severity is unchanged, or the symptoms description clearly serves to provide an overview of all relevant symptoms.
    B. Worsened: the severity the patient experiences of this symptom has gotten worse. This can also be indicated by a description that the patient has started experiencing the symptom recently.
    C. Improved: the severity the patient experiences of this symptom has gotten better/less. This can also be indicated by a description that the patient has recently experienced symptom opposites."""

OUTPUT = "{symptom-group-number} = {progression-label-letter}, for all symptom groups."


def get_1step_prompts(cat): 
    """
    Get all 1-step prompts for a certain category. 

    Parameters: 
    cat (int): category 

    Returns: 
    prompts (list[tuple[str, str]]): list of all prompts as a tuple of the prompt and the prompt id  
    """

    # base format template 
    base_prompt = dedent("""\
        {role}{overall_goal}{task}{patient_info}

        <start clinical note>
        {{{{}}}}
        <end clinical note>

        Choose from one of these options:
        {LABELS}

        Focus only on {{symptom}}, ignore other symptoms. 
        Take the entire note into account. 
        Ignore symptoms that are unrelated to the patient's depression. 
        Focus on relative progression. Both severe and mild symptoms can improve, worsen, or stay the same. 
        Assess the progression as relative to the current treatment trajectory: a symptom can be worse or better than before starting the treatment trajectory, this is unrelated to progression within the trajectory. Use the context of the note to assess if the progression took place during the treatment trajectory or before. 
        Do not make assumptions, only use what is written in the clinical note. 
        Do not give an explanation. Only write a single letter corresponding to the progression label that best describes the content of the note concerning {{symptom}}."""
    )

    # format all prompt variations 
    prompts = [
        (
            # insert prompt items 
            base_prompt.format(    
                task = task.text, 
                role = role.text, 
                overall_goal = overall_goal.text,  
                patient_info = patient_info.text, 
                LABELS = LABELS
            # insert correct symptom and symptom description 
            ).format(
                symptom = SYMPTOMS[cat][0],
                description = SYMPTOMS[cat][1]
            ),
            # format prompt id as a combination of prompt item descriptions 
            '|'.join([task.desc, role.desc, overall_goal.desc, patient_info.desc])
        )

        # for all combinations of prompt items 
        for task, role, overall_goal, patient_info in product(
            TASK, ROLE, OVERALL_GOAL, PATIENT_INFO
        )
    ]

    return prompts 

def get_2step_prompts(cat):
    """
    Get all 2-step prompts for a certain category. 

    Parameters: 
    cat (int): category 

    Returns: 
    prompts (list[tuple[tuple(str, str), str]]): list of all prompts as a tuple of: a tuple of the mention-prompt + the progression-prompt and and the (shared) prompt id  
    """

    # base format template for mention-prompt
    mention_prompt = dedent("""\
        {role}{task}{patient_info}

        <Start clinical note>
        {{{{}}}}
        <End clinical note>

        Focus only on {{symptom}}, ignore other symptoms. 
        Take the entire note into account. 
        Ignore symptoms that are unrelated to the patient's depression. 
        Do not make assumptions, only use what is written in the clinical note. 
        Do not give an explanation. Only write 'yes' or 'no'."""
    )

    # base format template for progression-prompt
    progression_prompt = dedent("""\
        {role}{overall_goal}{task}{patient_info}

        <start clinical note>
        {{{{}}}}
        <end clinical note>

        Choose from one of these options:
        {LABELS}

        Focus only on {{symptom}}, ignore other symptoms. 
        Take the entire note into account. 
        Ignore symptoms that are unrelated to the patient's depression. 
        Focus on relative progression. Both severe and mild symptoms can improve, worsen, or stay the same. 
        Assess the progression as relative to the current treatment trajectory: a symptom can be worse or better than before starting the treatment trajectory, this is unrelated to progression within the trajectory. Use the context of the note to assess if the progression took place during the treatment trajectory or before. 
        Do not make assumptions, only use what is written in the clinical note. 
        Do not give an explanation. Only write a single letter corresponding to the progression label that best describes the content of the note concerning {{symptom}}."""
    )

    # format all prompt variations 
    prompts = [
        (
            (
                #* mention prompt 
                # insert prompt items
                mention_prompt.format(
                    task = task.text['mention'], 
                    role = role.text, 
                    patient_info = patient_info.text
                # insert correct symptom and symptom description 
                ).format(
                    symptom = SYMPTOMS[cat][0],
                    description = SYMPTOMS[cat][1]
                ),
                
                #* progression prompt 
                # insert prompt items
                progression_prompt.format(
                    task = task.text['progression'], 
                    role = role.text, 
                    overall_goal = overall_goal.text,  
                    patient_info = patient_info.text, 
                    LABELS = LABELS_PROGRESSION
                # insert correct symptom and symptom description 
                ).format(
                    symptom = SYMPTOMS[cat][0],
                    description = SYMPTOMS[cat][1]
                    )
            ),
            # format prompt id as a combination of prompt item descriptions 
            '|'.join([task.desc, role.desc, overall_goal.desc, patient_info.desc])
        )

            # for all combinations of prompt items 
            for task, role, overall_goal, patient_info in product(
                TASK_2STEP, ROLE, OVERALL_GOAL, PATIENT_INFO
            )
    ]

    return prompts
