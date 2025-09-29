from functools import partial
from typing import Optional
import json

from tau2.data_model.tasks import Task
from tau2.domains.education.data_model import EducationDB
from tau2.domains.education.tools import EducationTools

from tau2.domains.education.utils import (
    EDUCATION_DB_PATH,
    EDUCATION_POLICY_PATH,
    EDUCATION_USER_POLICY_PATH,
    EDUCATION_TASK_SET_PATH,
    EDUCATION_DATA_DIR,
)
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolKitBase
from tau2.utils import load_file


def get_environment(
    db: Optional[EducationDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if db is None:
        db = EducationDB.load(EDUCATION_DB_PATH)
    tools = EducationTools(db)

    if solo_mode:
        policy = load_file(EDUCATION_USER_POLICY_PATH)
    else:
        policy = load_file(EDUCATION_POLICY_PATH)

    return Environment(
        domain_name="education",
        policy=policy,
        tools=tools,
    )


def load_personas() -> dict:
    """Load user personas from the education domain personas file"""
    try:
        personas_path = EDUCATION_DATA_DIR / "user_personas.json"
        with open(personas_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def inject_persona_data(task_data: dict, personas: dict) -> dict:
    """Inject persona data into task based on persona key"""
    if "user_scenario" in task_data and "persona" in task_data["user_scenario"]:
        persona_key = task_data["user_scenario"]["persona"]
        if persona_key in personas:
            task_data["user_scenario"]["persona"] = personas[persona_key]
    return task_data


def get_tasks() -> list[Task]:
    with open(EDUCATION_TASK_SET_PATH, "r") as fp:
        task_data_list = json.load(fp)

    # Load personas for injection
    personas = load_personas()

    tasks = []
    for task_data in task_data_list:
        # Inject persona data if available
        task_data = inject_persona_data(task_data, personas)
        tasks.append(Task(**task_data))

    return tasks
