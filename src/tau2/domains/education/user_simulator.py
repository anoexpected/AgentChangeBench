"""
Education-specific user simulator that uses domain policy as system prompt.
"""

from typing import Optional, Tuple

from loguru import logger

from tau2.data_model.message import (
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from tau2.data_model.tasks import UserInstructions
from tau2.environment.tool import Tool
from tau2.user.base import (
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    BaseUser,
    UserState,
    ValidUserInputMessage,
    is_valid_user_history_message,
)
from tau2.domains.education.utils import EDUCATION_USER_POLICY_PATH
from tau2.utils.llm_utils import generate
from tau2.utils import load_file


EDUCATION_USER_SYSTEM_PROMPT = """
{education_user_policy}

<scenario>
{instructions}
</scenario>
""".strip()


class EducationUserSimulator(BaseUser):
    """Education-specific user simulator that uses education user policy as system prompt."""

    def __init__(
        self,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[UserInstructions] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the education user simulator.

        Args:
            tools: List of tools available to the user
            instructions: User instructions for the scenario
            llm: LLM model to use
            llm_args: Additional arguments for the LLM
        """
        super().__init__(tools, instructions, llm, llm_args)
        self.education_user_policy = load_file(EDUCATION_USER_POLICY_PATH)

    def _get_system_message(self, instructions: UserInstructions) -> SystemMessage:
        """
        Get the system message with education user policy and scenario instructions.

        Args:
            instructions: User instructions for the scenario

        Returns:
            SystemMessage with education context and scenario
        """
        system_content = EDUCATION_USER_SYSTEM_PROMPT.format(
            education_user_policy=self.education_user_policy,
            instructions=instructions.format(),
        )
        return SystemMessage(content=system_content)

    def get_next_user_message(
        self,
        state: UserState,
        agent_message: Optional[Message] = None,
    ) -> Tuple[ValidUserInputMessage, UserState]:
        """
        Generate the next user message based on education domain context.

        Args:
            state: Current user state
            agent_message: Last message from the agent

        Returns:
            Tuple of (next user message, updated state)
        """
        logger.debug("Generating next education user message...")

        # Build conversation history
        conversation_history = []

        if state.instructions:
            system_msg = self._get_system_message(state.instructions)
            conversation_history.append(system_msg)

        # Add previous conversation messages
        for msg in state.history:
            if is_valid_user_history_message(msg):
                conversation_history.append(msg)

        # Add the latest agent message if provided
        if agent_message:
            conversation_history.append(agent_message)

        # Generate response using LLM
        try:
            response = generate(
                messages=conversation_history,
                tools=self.tools,
                model=self.llm,
                **self.llm_args,
            )

            logger.debug(f"Generated education user response: {response}")

            # Handle special control messages
            if response.content and response.content.strip() in [STOP, TRANSFER, OUT_OF_SCOPE]:
                return response.content.strip(), state

            # Handle tool calls (if user has tools)
            if isinstance(response, MultiToolMessage) and response.tool_calls:
                # Update state with new message
                new_state = UserState(
                    instructions=state.instructions,
                    history=state.history + [response],
                )
                return response, new_state

            # Handle regular user message
            if isinstance(response, UserMessage):
                # Update state with new message
                new_state = UserState(
                    instructions=state.instructions,
                    history=state.history + [response],
                )
                return response, new_state

            # Fallback for unexpected response types
            logger.warning(f"Unexpected response type in education user simulator: {type(response)}")
            fallback_msg = UserMessage(content="I need help with my academic planning.")
            new_state = UserState(
                instructions=state.instructions,
                history=state.history + [fallback_msg],
            )
            return fallback_msg, new_state

        except Exception as e:
            logger.error(f"Error generating education user message: {str(e)}")
            # Return a fallback message
            fallback_msg = UserMessage(content="I'm having trouble with my request. Can you help me?")
            return fallback_msg, state

