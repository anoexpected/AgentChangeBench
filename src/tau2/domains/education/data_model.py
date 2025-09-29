import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field
from tau2.domains.education.utils import EDUCATION_DB_PATH
from tau2.environment.db import DB
from tau2.utils.pydantic_utils import BaseModelNoExtra


DEFAULT_START_DATE = datetime.date(2025, 1, 1)


class AcademicStanding(str, Enum):
    GOOD_STANDING = "good_standing"
    ACADEMIC_WARNING = "academic_warning"
    ACADEMIC_PROBATION = "academic_probation"
    ACADEMIC_DISMISSAL = "academic_dismissal"


class StudentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    GRADUATED = "graduated"
    WITHDRAWN = "withdrawn"


class CourseStatus(str, Enum):
    AVAILABLE = "available"
    FULL = "full"
    CANCELLED = "cancelled"
    WAITLIST = "waitlist"


class SemesterType(str, Enum):
    FALL = "fall"
    SPRING = "spring"
    SUMMER = "summer"


class Student(BaseModelNoExtra):
    student_id: str = Field(description="Unique student identifier")
    name: str = Field(description="Full student name")
    email: str = Field(description="University email address")
    major: Optional[str] = Field(None, description="Current major")
    minor: Optional[str] = Field(None, description="Current minor")
    academic_standing: AcademicStanding = Field(AcademicStanding.GOOD_STANDING, description="Academic standing")
    status: StudentStatus = Field(StudentStatus.ACTIVE, description="Enrollment status")
    gpa: Optional[float] = Field(None, description="Current GPA")
    total_credits: int = Field(0, description="Total credits earned")
    enrollment_date: datetime.date = Field(DEFAULT_START_DATE, description="Enrollment date")
    advisor_id: Optional[str] = Field(None, description="Assigned advisor ID")


class Course(BaseModelNoExtra):
    course_id: str = Field(description="Unique course identifier (e.g., CS101)")
    title: str = Field(description="Course title")
    subject: str = Field(description="Subject area (e.g., CS, MATH)")
    level: int = Field(description="Course level (100, 200, etc.)")
    credits: int = Field(description="Credit hours")
    prerequisites: List[str] = Field(default_factory=list, description="Required prerequisite courses")
    corequisites: List[str] = Field(default_factory=list, description="Required corequisite courses")
    description: str = Field(description="Course description")
    instructor: Optional[str] = Field(None, description="Instructor name")
    max_enrollment: int = Field(description="Maximum students allowed")
    current_enrollment: int = Field(0, description="Currently enrolled students")
    status: CourseStatus = Field(CourseStatus.AVAILABLE, description="Course availability status")
    waitlist_count: int = Field(0, description="Number of students on waitlist")


class Enrollment(BaseModelNoExtra):
    enrollment_id: str = Field(description="Unique enrollment identifier")
    student_id: str = Field(description="Student ID")
    course_id: str = Field(description="Course ID")
    semester: SemesterType = Field(description="Semester enrolled")
    year: int = Field(description="Academic year")
    grade: Optional[str] = Field(None, description="Final grade (A, B, C, D, F, W)")
    status: str = Field("enrolled", description="Enrollment status")


class Major(BaseModelNoExtra):
    major_id: str = Field(description="Unique major identifier")
    name: str = Field(description="Major name")
    department: str = Field(description="Department offering the major")
    required_credits: int = Field(description="Total credits required for major")
    required_courses: List[str] = Field(default_factory=list, description="Required course IDs")
    elective_credits: int = Field(0, description="Elective credits required")
    minimum_gpa: float = Field(2.0, description="Minimum GPA requirement")


class FinancialAid(BaseModelNoExtra):
    aid_id: str = Field(description="Unique aid identifier")
    student_id: str = Field(description="Student ID")
    aid_type: str = Field(description="Type of aid (grant, loan, scholarship)")
    amount: float = Field(description="Aid amount")
    semester: SemesterType = Field(description="Semester")
    year: int = Field(description="Academic year")
    disbursed: bool = Field(False, description="Whether aid has been disbursed")
    requirements: List[str] = Field(default_factory=list, description="Aid requirements")


class Advisor(BaseModelNoExtra):
    advisor_id: str = Field(description="Unique advisor identifier")
    name: str = Field(description="Advisor name")
    email: str = Field(description="Advisor email")
    department: str = Field(description="Department")
    office_location: str = Field(description="Office location")
    office_hours: str = Field(description="Office hours")
    specializations: List[str] = Field(default_factory=list, description="Areas of specialization")


class CampusResource(BaseModelNoExtra):
    resource_id: str = Field(description="Unique resource identifier")
    name: str = Field(description="Resource name")
    type: str = Field(description="Resource type (library, lab, service)")
    location: str = Field(description="Physical location")
    hours: str = Field(description="Operating hours")
    description: str = Field(description="Resource description")
    contact_info: str = Field(description="Contact information")
    services: List[str] = Field(default_factory=list, description="Available services")


class AcademicCalendar(BaseModelNoExtra):
    event_id: str = Field(description="Unique event identifier")
    event_name: str = Field(description="Event name")
    event_type: str = Field(description="Event type (deadline, holiday, etc.)")
    date: datetime.date = Field(description="Event date")
    semester: SemesterType = Field(description="Applicable semester")
    year: int = Field(description="Academic year")
    description: str = Field(description="Event description")


class Transcript(BaseModelNoExtra):
    student_id: str = Field(description="Student ID")
    enrollments: List[Enrollment] = Field(default_factory=list, description="Course enrollments")
    gpa: float = Field(description="Current GPA")
    total_credits: int = Field(description="Total credits earned")
    academic_standing: AcademicStanding = Field(description="Current academic standing")
    honors: List[str] = Field(default_factory=list, description="Academic honors")


class EducationDB(DB):
    students: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="All students")
    courses: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="All courses")
    campus_resources: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="All campus resources")
    faculty: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Faculty information")
    academic_records: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Academic records")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Database metadata")

    def get_data(self) -> Dict[str, Any]:
        """Get all data as a dictionary."""
        result = {
            "students": self.students,
            "courses": self.courses,
            "campus_resources": self.campus_resources,
        }
        if self.faculty:
            result["faculty"] = self.faculty
        if self.academic_records:
            result["academic_records"] = self.academic_records
        if self.metadata:
            result["metadata"] = self.metadata
        return result


def get_db() -> EducationDB:
    """Get the education database."""
    return EducationDB.load(EDUCATION_DB_PATH)

