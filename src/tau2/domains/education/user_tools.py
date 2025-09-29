"""User tools for the education domain."""

from typing import Optional
from loguru import logger

from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class EducationUserTools(ToolKitBase):
    """User tools for the education domain."""

    def __init__(self, db=None) -> None:
        super().__init__(db)
        self._user_state = {
            "authenticated": False,
            "student_id": None,
            "email": None,
        }

    @is_tool(tool_type=ToolType.WRITE)
    def update_user(
        self,
        authenticated: Optional[bool] = None,
        student_id: Optional[str] = None,
        email: Optional[str] = None,
    ) -> str:
        """
        Update user authentication state and information.
        
        Args:
            authenticated: Whether the user is authenticated
            student_id: Student ID
            email: Student email address
            
        Returns:
            Status message
        """
        try:
            updates = []
            
            if authenticated is not None:
                self._user_state["authenticated"] = authenticated
                updates.append(f"authentication status: {authenticated}")
            
            if student_id is not None:
                self._user_state["student_id"] = student_id
                updates.append(f"student ID: {student_id}")
            
            if email is not None:
                self._user_state["email"] = email
                updates.append(f"email: {email}")
            
            if updates:
                return f"User information updated: {', '.join(updates)}"
            else:
                return "No user information provided to update"
            
        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            return f"Error updating user information: {str(e)}"

    def get_user_state(self) -> dict:
        """Get current user state."""
        return self._user_state.copy()

    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self._user_state.get("authenticated", False)

    def get_student_id(self) -> Optional[str]:
        """Get current student ID."""
        return self._user_state.get("student_id")

    def get_email(self) -> Optional[str]:
        """Get current email."""
        return self._user_state.get("email")

