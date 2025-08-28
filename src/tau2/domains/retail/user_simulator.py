"""
Retail-specific user simulator that uses domain policy as system prompt.
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
from tau2.domains.retail.utils import RETAIL_USER_POLICY_PATH
from tau2.utils.llm_utils import generate
from tau2.utils import load_file


RETAIL_USER_SYSTEM_PROMPT = """
{retail_user_policy}

<scenario>
{instructions}
</scenario>
""".strip()


class RetailUserSimulator(BaseUser):
    """Retail-specific user simulator that uses retail user policy as system prompt."""

    def __init__(
        self,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[UserInstructions] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(instructions=instructions, llm=llm, llm_args=llm_args)
        self.tools = tools

    @property
    def retail_user_policy(self) -> str:
        """
        Load the retail user policy from file.
        The policy includes tool handling and goal shift capabilities.
        """
        return load_file(RETAIL_USER_POLICY_PATH)

    @property
    def system_prompt(self) -> str:
        """
        The system prompt for the retail user simulator.
        Uses retail user policy which includes tool handling and goal shift capabilities.
        """
        if self.instructions is None:
            logger.warning("No instructions provided for retail user simulator")

        system_prompt = RETAIL_USER_SYSTEM_PROMPT.format(
            retail_user_policy=self.retail_user_policy,
            instructions=self.instructions,
        )

        return system_prompt

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        """
        Get the initial state of the retail user simulator.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_user_history_message(m) for m in message_history), (
            "Invalid user message history. User messages must be of type UserMessage, AssistantMessage, or ToolMessage to User."
        )

        user_state = UserState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
        return user_state

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        """
        Check if the message is a stop message.
        """
        if message.is_tool_call():
            return False
        assert message.content is not None
        return (
            STOP in message.content
            or TRANSFER in message.content
            or OUT_OF_SCOPE in message.content
        )

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        return self._generate_next_message(message, state)

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """Get the response from the retail user simulator.

        Args:
            message: The assistant or tool message.
            state: The current user state.

        Returns:
            The user response and updated state.
        """
        # Generate response using LLM
        try:
            # Update state with incoming message first
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)

            # Flip roles so the LLM responds as assistant to the user's perspective
            messages_for_llm = state.system_messages + state.flip_roles()

            assistant_message = generate(
                model=self.llm,
                messages=messages_for_llm,
                tools=self.tools,
                **(self.llm_args or {}),
            )

            # Ensure non-empty content when there are no tool calls
            try:
                has_tool_calls = bool(getattr(assistant_message, "tool_calls", None))
                content_value = getattr(assistant_message, "content", None)
                if (not has_tool_calls) and (content_value is None or (isinstance(content_value, str) and content_value.strip() == "")):
                    assistant_message.content = "(no content)"
            except Exception:
                pass

            # Build user message from assistant reply
            user_message = UserMessage(
                role="user",
                content=assistant_message.content,
                cost=assistant_message.cost,
                usage=assistant_message.usage,
                raw_data=assistant_message.raw_data,
            )

            # Flip any tool calls so they are attributed to the user
            if getattr(assistant_message, "tool_calls", None):
                user_message.tool_calls = []
                for tool_call in assistant_message.tool_calls:
                    user_message.tool_calls.append(
                        ToolCall(
                            id=tool_call.id,
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                            requestor="user",
                        )
                    )

            # Update and return state
            state.messages.append(user_message)
            return user_message, state

        except Exception as e:
            logger.error(f"Error generating user response: {e}")
            # Fallback response
            user_message = UserMessage(
                role="user",
                content="I'm having trouble responding. Could you please repeat that?",
            )
            updated_state = UserState(
                system_messages=state.system_messages,
                messages=state.messages + [message, user_message],
            )
            return user_message, updated_state
