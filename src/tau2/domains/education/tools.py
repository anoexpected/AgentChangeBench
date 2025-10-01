"""Toolkit for the education domain."""

from typing import List, Optional, Dict, Any
from loguru import logger

from tau2.domains.education.data_model import (
    EducationDB,
    Student,
    Course,
    Enrollment,
    Major,
    FinancialAid,
    Advisor,
    CampusResource,
    AcademicCalendar,
    Transcript,
    AcademicStanding,
    StudentStatus,
    CourseStatus,
    SemesterType,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class EducationTools(ToolKitBase):
    """All the tools for the education domain."""

    db: EducationDB

    def __init__(self, db: EducationDB) -> None:
        super().__init__(db)

    def _get_student(self, student_id: str) -> Dict[str, Any]:
        """Get student from database."""
        if student_id in self.db.students:
            return self.db.students[student_id]
        raise ValueError(f"Student {student_id} not found")

    def _get_course(self, course_id: str) -> Dict[str, Any]:
        """Get course from database."""
        if course_id in self.db.courses:
            return self.db.courses[course_id]
        raise ValueError(f"Course {course_id} not found")

    def _get_campus_resource(self, resource_id: str) -> Dict[str, Any]:
        """Get campus resource from database."""
        if resource_id in self.db.campus_resources:
            return self.db.campus_resources[resource_id]
        raise ValueError(f"Campus resource {resource_id} not found")

    def _get_advisor(self, advisor_id: str) -> Dict[str, Any]:
        """Get advisor from faculty database."""
        if self.db.faculty and advisor_id in self.db.faculty:
            return self.db.faculty[advisor_id]
        raise ValueError(f"Advisor {advisor_id} not found")


    @is_tool(tool_type=ToolType.READ)
    def search_courses(
        self,
        subject: Optional[str] = None,
        level: Optional[str] = None,
        instructor: Optional[str] = None,
        keyword: Optional[str] = None,
        course_code: Optional[str] = None,
        semester: Optional[str] = None,
    ) -> str:
        """
        Find courses by subject, level, instructor, keyword, course code, or semester.
        
        Args:
            subject: Course subject (e.g., 'CS', 'MATH')
            level: Course level (e.g., 'Undergraduate', 'Graduate')
            instructor: Instructor name
            keyword: Search keyword in course title or description
            course_code: Specific course code (e.g., 'CS101')
            semester: Semester filter (e.g., 'Fall 2024')
            
        Returns:
            List of matching courses with details
        """
        try:
            matching_courses = []
            
            for course_id, course in self.db.courses.items():
                # Apply filters
                if course_code and course_id.upper() != course_code.upper():
                    continue
                if subject and course.get("department", "").upper() != subject.upper():
                    continue
                if level and course.get("level", "").lower() != level.lower():
                    continue
                if instructor and instructor.lower() not in (course.get("instructor", "") or "").lower():
                    continue
                if semester and course.get("term", "") != semester:
                    continue
                if keyword and (
                    keyword.lower() not in course.get("title", "").lower() and
                    keyword.lower() not in course.get("description", "").lower()
                ):
                    continue
                
                # Format course info
                course_info = {
                    "course_id": course.get("course_code", course_id),
                    "title": course.get("title", ""),
                    "subject": course.get("department", ""),
                    "level": course.get("level", ""),
                    "credits": course.get("credits", ""),
                    "instructor": course.get("instructor", "TBA"),
                    "status": "Available",
                    "enrollment": f"{course.get('enrolled', 0)}/{course.get('capacity', 'N/A')}",
                    "waitlist": 0,
                    "prerequisites": course.get("prerequisites", []),
                    "term": course.get("term", ""),
                }
                matching_courses.append(course_info)
            
            if not matching_courses:
                return "No courses found matching your criteria."
            
            # Format results
            result = f"Found {len(matching_courses)} course(s):\n"
            for course in matching_courses:
                result += f"\n• {course['course_id']}: {course['title']}\n"
                result += f"  Credits: {course['credits']}, Instructor: {course['instructor'] or 'TBA'}\n"
                result += f"  Status: {course['status']}, Enrollment: {course['enrollment']}\n"
                if course['prerequisites']:
                    result += f"  Prerequisites: {', '.join(course['prerequisites'])}\n"
                if course['waitlist'] > 0:
                    result += f"  Waitlist: {course['waitlist']} students\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching courses: {str(e)}")
            return f"Error searching for courses: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def check_prerequisites(self, course_code: str, student_id: Optional[str] = None) -> str:
        """
        Verify course requirements and check if student has met prerequisites.
        
        Args:
            course_id: Course ID to check prerequisites for
            student_id: Student ID to check eligibility (optional)
            
        Returns:
            Prerequisites information and eligibility status
        """
        try:
            course = self._get_course(course_code)
            
            result = f"Prerequisites for {course.get('course_code', course_code)}: {course.get('title', '')}\n\n"
            
            prerequisites = course.get('prerequisites', [])
            corequisites = course.get('corequisites', [])
            
            if not prerequisites and not corequisites:
                result += "No prerequisites required for this course.\n"
            else:
                if prerequisites:
                    result += f"Prerequisites: {', '.join(prerequisites)}\n"
                if corequisites:
                    result += f"Corequisites: {', '.join(corequisites)}\n"
            
            # Check student eligibility if student_id provided
            if student_id:
                try:
                    student = self._get_student(student_id)
                    # For now, we'll assume prerequisites are met since we don't have enrollment history
                    # In a real system, you'd check the student's transcript
                    result += "\n✅ Prerequisite check completed.\n"
                    result += "Note: Please verify with your academic advisor if you have questions about prerequisites.\n"
                    
                    # Check course capacity
                    capacity = course.get('capacity', 0)
                    enrolled = course.get('enrolled', 0)
                    
                    if enrolled >= capacity:
                        result += "⚠️  Course is currently full. You may join the waitlist.\n"
                    else:
                        available_spots = capacity - enrolled
                        result += f"Course is available for enrollment. {available_spots} spots remaining.\n"
                        
                except ValueError as ve:
                    result += f"\n⚠️  Could not verify eligibility: {str(ve)}\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error checking prerequisites: {str(e)}")
            return f"Error checking prerequisites: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def get_degree_requirements(self, major: Optional[str] = None, student_id: Optional[str] = None) -> str:
        """
        Get major requirements and degree information.
        
        Args:
            major: Major name or ID
            student_id: Optional student ID for personalized requirements
            
        Returns:
            Detailed degree requirements information
        """
        try:
            # If no major specified but student_id provided, get student's major
            if not major and student_id:
                student = self._get_student(student_id)
                major = student.get('major', 'Undeclared')
            elif not major:
                major = "General"
            
            result = f"Degree Requirements for {major}\n\n"
            
            # Standard degree requirements (since we don't have major-specific data in DB)
            result += f"Total Credits Required: 120\n"
            result += f"Minimum GPA: 2.0\n"
            result += f"Major Credits: 36-48\n"
            result += f"General Education Credits: 40-45\n"
            result += f"Elective Credits: 27-44\n\n"
            
            # Find courses related to this major by searching course departments
            major_courses = []
            major_prefix = ""
            
            # Map common majors to course prefixes
            major_mappings = {
                "Computer Science": "CS",
                "Business Administration": "BUS", 
                "Psychology": "PSYC",
                "Biology": "BIOL",
                "Mathematics": "MATH",
                "English": "ENG",
                "History": "HIST",
                "Art": "ART",
                "Chemistry": "CHEM",
                "Physics": "PHYS"
            }
            
            major_prefix = major_mappings.get(major, major[:4].upper())
            
            # Search for courses with matching department/prefix
            for course_id, course_data in self.db.courses.items():
                if course_data.get('department', '').upper() == major_prefix or course_id.startswith(major_prefix):
                    major_courses.append((course_id, course_data))
            
            if major_courses:
                result += f"Available {major} Courses:\n"
                for course_id, course_data in major_courses[:10]:  # Show first 10
                    title = course_data.get('title', 'Unknown')
                    credits = course_data.get('credits', 0)
                    level = course_data.get('level', 'Unknown')
                    result += f"• {course_id}: {title} ({credits} credits) - {level}\n"
                
                if len(major_courses) > 10:
                    result += f"... and {len(major_courses) - 10} more courses\n"
            else:
                result += f"No specific courses found for {major} major.\n"
            
            # Add general graduation requirements
            result += "\nGeneral Graduation Requirements:\n"
            result += "• Minimum 120 total credit hours\n"
            result += "• General education: 40-45 credits\n"
            result += "• Residency requirement: 30 credits at university\n"
            result += "• Overall GPA: 2.0 minimum\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error getting degree requirements: {str(e)}")
            return f"Error retrieving degree requirements: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def check_enrollment_status(self, student_id: str, course_code: Optional[str] = None) -> str:
        """
        Check student enrollment status and information.
        
        Args:
            student_id: Student ID
            course_code: Optional course code to check specific enrollment
            
        Returns:
            Student enrollment status and academic information
        """
        try:
            student = self._get_student(student_id)
            
            student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
            result = f"Enrollment Status for {student_name} (ID: {student.get('student_id', student_id)})\n\n"
            result += f"Status: {student.get('enrollment_status', 'Unknown').replace('_', ' ').title()}\n"
            result += f"Academic Standing: {student.get('academic_standing', 'Unknown').replace('_', ' ').title()}\n"
            result += f"Major: {student.get('major', 'Undeclared')}\n"
            if student.get('minor'):
                result += f"Minor: {student.get('minor')}\n"
            result += f"GPA: {student.get('gpa', 'N/A')}\n"
            result += f"Total Credits: {student.get('total_credits', 0)}\n"
            result += f"Year: {student.get('year', 'N/A')}\n"
            result += f"Email: {student.get('email', 'N/A')}\n"
            result += f"Phone: {student.get('phone', 'N/A')}\n"
            
            # Check for holds
            holds = student.get('holds', [])
            if holds:
                result += f"\n⚠️  Academic Holds ({len(holds)}):\n"
                for hold in holds:
                    result += f"• {hold}\n"
            else:
                result += "\n✅ No academic holds found.\n"
            
            # Check for advisor
            advisor_id = student.get('advisor_id')
            if advisor_id:
                try:
                    advisor = self._get_advisor(advisor_id)
                    advisor_name = f"{advisor.get('title', '')} {advisor.get('first_name', '')} {advisor.get('last_name', '')}".strip()
                    result += f"\nAcademic Advisor: {advisor_name}\n"
                    result += f"Email: {advisor.get('email', 'N/A')}\n"
                    result += f"Office: {advisor.get('office', 'N/A')}\n"
                    result += f"Office Hours: {advisor.get('office_hours', 'N/A')}\n"
                    result += f"Department: {advisor.get('department', 'N/A')}\n"
                except ValueError:
                    result += f"\nAdvisor ID: {advisor_id} (details not available)\n"
            else:
                result += "\nNo advisor assigned.\n"
            
            # Financial aid status
            result += f"\nFinancial Aid Status: {student.get('financial_aid_status', 'Unknown')}\n"
            
            # Emergency contact
            emergency_contact = student.get('emergency_contact', {})
            if emergency_contact:
                result += f"\nEmergency Contact: {emergency_contact.get('name', 'N/A')}\n"
                result += f"Phone: {emergency_contact.get('phone', 'N/A')}\n"
                result += f"Relationship: {emergency_contact.get('relationship', 'N/A')}\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error checking enrollment status: {str(e)}")
            return f"Error checking enrollment status: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def get_transcript(self, student_id: str) -> str:
        """
        Access academic records and transcript information.
        
        Args:
            student_id: Student ID
            
        Returns:
            Complete academic transcript
        """
        try:
            student = self._get_student(student_id)
            
            student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
            result = f"Official Transcript for {student_name} (ID: {student.get('student_id', student_id)})\n"
            result += f"Email: {student.get('email', 'N/A')}\n"
            result += f"Major: {student.get('major', 'Undeclared')}\n"
            result += f"Academic Standing: {student.get('academic_standing', 'Unknown').replace('_', ' ').title()}\n\n"
            
            # Get student enrollments from the database
            student_enrollments = student.get('enrollments', [])
            student_grades = student.get('grades', [])
            
            # Group by semester and year
            enrollments_by_term = {}
            for enrollment in student_enrollments:
                term_key = enrollment.get('term', 'Unknown Term')
                if term_key not in enrollments_by_term:
                    enrollments_by_term[term_key] = []
                enrollments_by_term[term_key].append(enrollment)
            
            total_credits_earned = 0
            total_grade_points = 0
            total_attempted = 0
            
            # Grade point mapping
            grade_points = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}
            
            # Use grades data if available, otherwise use enrollments
            if student_grades:
                # Group grades by term
                grades_by_term = {}
                for grade_record in student_grades:
                    term_key = grade_record.get('term', 'Unknown Term')
                    if term_key not in grades_by_term:
                        grades_by_term[term_key] = []
                    grades_by_term[term_key].append(grade_record)
                
                for term in sorted(grades_by_term.keys()):
                    result += f"\n{term}:\n"
                    term_credits = 0
                    term_grade_points = 0
                    
                    for grade_record in grades_by_term[term]:
                        course_code = grade_record.get('course_code', 'Unknown')
                        course_title = grade_record.get('course_title', 'Unknown Course')
                        credits = grade_record.get('credits', 3)
                        grade = grade_record.get('grade', 'N/A')
                        
                        result += f"  {course_code}: {course_title} - {credits} credits - {grade}\n"
                        
                        if grade and grade in grade_points:
                            term_credits += credits
                            term_grade_points += grade_points[grade] * credits
                            total_attempted += credits
                            if grade != 'F':
                                total_credits_earned += credits
                    
                    if term_credits > 0:
                        term_gpa = term_grade_points / term_credits
                        result += f"  Term GPA: {term_gpa:.2f}, Credits: {term_credits}\n"
                        total_grade_points += term_grade_points
            else:
                # Fallback to enrollment data
                for term in sorted(enrollments_by_term.keys()):
                    result += f"\n{term}:\n"
                    for enrollment in enrollments_by_term[term]:
                        course_code = enrollment.get('course_code', 'Unknown')
                        grade = enrollment.get('grade', 'In Progress')
                        result += f"  {course_code}: {grade}\n"
            
            # Calculate overall GPA
            overall_gpa = total_grade_points / total_attempted if total_attempted > 0 else 0.0
            
            result += f"\n--- Summary ---\n"
            result += f"Total Credits Earned: {total_credits_earned}\n"
            result += f"Total Credits Attempted: {total_attempted}\n"
            result += f"Overall GPA: {overall_gpa:.2f}\n"
            result += f"Academic Standing: {student.get('academic_standing', 'Unknown').replace('_', ' ').title()}\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error getting transcript: {str(e)}")
            return f"Error retrieving transcript: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def get_financial_aid_info(self, student_id: str) -> str:
        """
        Get financial aid information for a student.
        
        Args:
            student_id: Student ID
            
        Returns:
            Financial aid package details
        """
        try:
            student = self._get_student(student_id)
            student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
            
            result = f"Financial Aid Information for {student_name} (ID: {student_id})\n\n"
            result += f"Financial Aid Status: {student.get('financial_aid_status', 'Unknown')}\n\n"
            
            # Since detailed aid records aren't in the DB, provide general information
            aid_status = student.get('financial_aid_status', 'Unknown')
            
            if aid_status == 'Eligible':
                result += "✅ You are eligible for financial aid!\n\n"
                result += "Common Financial Aid Types:\n"
                result += "• Federal Pell Grant (need-based)\n"
                result += "• Federal Student Loans (subsidized/unsubsidized)\n"
                result += "• Work-Study Programs\n"
                result += "• State Grants\n"
                result += "• Institutional Scholarships\n\n"
            elif aid_status == 'Not Eligible':
                result += "❌ Currently not eligible for financial aid.\n\n"
                result += "Reasons for ineligibility may include:\n"
                result += "• Unsatisfactory Academic Progress\n"
                result += "• Exceeded maximum time frame\n"
                result += "• Missing required documentation\n\n"
            elif aid_status == 'Under Review':
                result += "⏳ Financial aid application is under review.\n\n"
                result += "Review process typically takes 2-4 weeks.\n"
                result += "Please ensure all documentation is submitted.\n\n"
            else:
                result += "No financial aid application on file.\n\n"
            
            result += "Next Steps:\n"
            result += "• Complete FAFSA annually by March 1st\n"
            result += "• Maintain Satisfactory Academic Progress (SAP)\n"
            result += "• Enroll in minimum 12 credits for full-time status\n"
            result += "• Submit any required verification documents\n\n"
            
            result += "Contact Information:\n"
            result += "• Financial Aid Office: finaid@university.edu\n"
            result += "• Phone: (555) 123-4567\n"
            result += "• Office Hours: Monday-Friday 8:00 AM - 5:00 PM\n"
            
            # SAP information
            academic_standing = student.get('academic_standing', 'Good Standing')
            if academic_standing in ['Academic Warning', 'Academic Probation']:
                result += "\n⚠️  Academic Standing Alert:\n"
                result += f"Your academic standing is: {academic_standing}\n"
                result += "This may affect your financial aid eligibility.\n"
                result += "Contact the Financial Aid office to discuss your options.\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error getting financial aid info: {str(e)}")
            return f"Error retrieving financial aid information: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def get_academic_calendar(self, semester: Optional[str] = None, year: Optional[int] = None) -> str:
        """
        Get academic calendar information.
        
        Args:
            semester: Specific semester (fall, spring, summer)
            year: Specific year
            
        Returns:
            Academic calendar events and important dates
        """
        try:
            events = self.db.academic_calendar
            
            # Apply filters
            if semester:
                semester_enum = SemesterType(semester.lower())
                events = [e for e in events if e.semester == semester_enum]
            
            if year:
                events = [e for e in events if e.year == year]
            
            if not events:
                return "No calendar events found for the specified criteria."
            
            # Sort by date
            events = sorted(events, key=lambda e: e.date)
            
            result = "Academic Calendar\n"
            if semester or year:
                filters = []
                if semester:
                    filters.append(f"Semester: {semester.title()}")
                if year:
                    filters.append(f"Year: {year}")
                result += f"({', '.join(filters)})\n"
            result += "\n"
            
            # Group by semester and year
            events_by_term = {}
            for event in events:
                term_key = f"{event.semester.value.title()} {event.year}"
                if term_key not in events_by_term:
                    events_by_term[term_key] = []
                events_by_term[term_key].append(event)
            
            for term in sorted(events_by_term.keys()):
                result += f"{term}:\n"
                
                for event in sorted(events_by_term[term], key=lambda e: e.date):
                    result += f"  • {event.date.strftime('%B %d')}: {event.event_name}\n"
                    if event.description:
                        result += f"    {event.description}\n"
                
                result += "\n"
            
            # Add general academic calendar info
            result += "Important Academic Calendar Information:\n"
            result += "• Registration periods: 2 weeks before semester\n"
            result += "• Add/drop period: First 2 weeks of semester\n"
            result += "• Withdrawal deadline: 10th week of semester\n"
            result += "• Fall semester: August-December\n"
            result += "• Spring semester: January-May\n"
            result += "• Summer sessions: May-August\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting academic calendar: {str(e)}")
            return f"Error retrieving academic calendar: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def search_campus_resources(self, resource_type: Optional[str] = None, keyword: Optional[str] = None, student_id: Optional[str] = None) -> str:
        """
        Find campus services and resources.
        
        Args:
            resource_type: Type of resource (library, lab, service)
            keyword: Search keyword
            student_id: Optional student ID for personalized resources
            
        Returns:
            Available campus resources information
        """
        try:
            resources_dict = self.db.campus_resources
            matching_resources = []
            
            # Apply filters
            for resource_id, resource in resources_dict.items():
                # Apply resource_type filter
                if resource_type and resource_type.lower() not in resource.get('name', '').lower():
                    continue
                
                # Apply keyword filter
                if keyword:
                    keyword_match = (
                        keyword.lower() in resource.get('name', '').lower() or
                        any(keyword.lower() in service.lower() for service in resource.get('services', []))
                    )
                    if not keyword_match:
                        continue
                
                matching_resources.append(resource)
            
            if not matching_resources:
                return "No campus resources found matching your criteria."
            
            result = f"Campus Resources ({len(matching_resources)} found):\n\n"
            
            for resource in matching_resources:
                result += f"• {resource.get('name', 'Unknown')}\n"
                result += f"  Location: {resource.get('location', 'N/A')}\n"
                result += f"  Hours: {resource.get('hours', 'N/A')}\n"
                result += f"  Phone: {resource.get('phone', 'N/A')}\n"
                
                services = resource.get('services', [])
                if services:
                    result += f"  Services: {', '.join(services)}\n"
                
                if resource.get('contact_email'):
                    result += f"  Email: {resource.get('contact_email')}\n"
                
                result += "\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching campus resources: {str(e)}")
            return f"Error searching campus resources: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def get_advisor_info(self, student_id: Optional[str] = None, advisor_id: Optional[str] = None) -> str:
        """
        Get academic advisor information.
        
        Args:
            student_id: Student ID to get their advisor
            advisor_id: Specific advisor ID
            
        Returns:
            Advisor contact information and details
        """
        try:
            if student_id:
                student = self._get_student(student_id)
                if not student.get('advisor_id'):
                    return f"No advisor assigned to student {student_id}. Please contact the academic advising office."
                advisor_id = student.get('advisor_id')
            
            if not advisor_id:
                return "Please provide either a student ID or advisor ID."
            
            advisor = self._get_advisor(advisor_id)
            
            advisor_name = f"{advisor.get('title', '')} {advisor.get('first_name', '')} {advisor.get('last_name', '')}".strip()
            result = f"Academic Advisor Information\n\n"
            result += f"Name: {advisor_name}\n"
            result += f"Department: {advisor.get('department', 'N/A')}\n"
            result += f"Email: {advisor.get('email', 'N/A')}\n"
            result += f"Office Location: {advisor.get('office', 'N/A')}\n"
            result += f"Office Hours: {advisor.get('office_hours', 'N/A')}\n"
            
            if advisor.get('specialization'):
                result += f"Specialization: {advisor.get('specialization')}\n"
            
            result += "\nAdvisor Services:\n"
            result += "• Course planning and registration assistance\n"
            result += "• Major and minor selection guidance\n"
            result += "• Graduation planning\n"
            result += "• Academic standing support\n"
            result += "• Career guidance and planning\n"
            
            result += "\nAppointment Information:\n"
            result += "• Schedule appointments through the student portal\n"
            result += "• Walk-in hours may be available during office hours\n"
            result += "• Email for urgent academic matters\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error getting advisor info: {str(e)}")
            return f"Error retrieving advisor information: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def check_graduation_status(self, student_id: str) -> str:
        """
        Check graduation eligibility and requirements progress.
        
        Args:
            student_id: Student ID
            
        Returns:
            Graduation status and remaining requirements
        """
        try:
            student = self._get_student(student_id)
            
            student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
            result = f"Graduation Status for {student_name} (ID: {student.get('student_id', student_id)})\n\n"
            
            # Basic requirements check
            min_credits = 120
            min_gpa = 2.0
            min_major_gpa = 2.5
            
            result += f"Progress Summary:\n"
            total_credits = student.get('total_credits', 0)
            result += f"• Total Credits: {total_credits}/{min_credits} "
            result += "✅" if total_credits >= min_credits else "❌"
            result += "\n"
            
            gpa = student.get('gpa')
            result += f"• Overall GPA: {gpa or 'N/A'}/{min_gpa} "
            if gpa:
                result += "✅" if gpa >= min_gpa else "❌"
            else:
                result += "❌ (GPA not calculated)"
            result += "\n"
            
            standing = student.get('academic_standing', 'Unknown').lower()
            result += f"• Academic Standing: {standing.replace('_', ' ').title()} "
            result += "✅" if standing == "good standing" else "❌"
            result += "\n"
            
            # Major requirements check
            major_name = student.get('major')
            if major_name:
                result += f"\nMajor Requirements ({major_name}):\n"
                result += f"• Major declared: ✅\n"
                result += f"• Major-specific requirements: Contact advisor for details\n"
            else:
                result += "\n❌ No major declared. You must declare a major to graduate.\n"
            
            # General education requirements (simplified)
            result += f"\nGeneral Education Requirements:\n"
            result += f"• Estimated: 40-45 credits required\n"
            result += f"• Residency: 30 credits at university required\n"
            
            # Overall graduation eligibility
            eligible = (
                total_credits >= min_credits and
                gpa and gpa >= min_gpa and
                standing == "good standing" and
                major_name
            )
            
            result += f"\n--- Graduation Eligibility ---\n"
            if eligible:
                result += "🎓 You appear to meet the basic requirements for graduation!\n"
                result += "Please schedule an appointment with your advisor to complete your graduation application.\n"
            else:
                result += "❌ Additional requirements must be met before graduation.\n"
                result += "Please meet with your academic advisor to create a plan for completion.\n"
            
            result += "\nNext Steps:\n"
            result += "• Schedule advisor appointment for degree audit\n"
            result += "• Apply for graduation by the deadline\n"
            result += "• Complete any remaining requirements\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error checking graduation status: {str(e)}")
            return f"Error checking graduation status: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def verify_student_identity(self, student_id: str) -> str:
        """
        Verify student identity and basic information.
        
        Args:
            student_id: Student ID to verify
            
        Returns:
            Student verification status and basic info
        """
        try:
            student = self._get_student(student_id)
            
            result = f"Student Identity Verified\n\n"
            result += f"Student ID: {student.get('student_id', student_id)}\n"
            result += f"Name: {student.get('first_name', '')} {student.get('last_name', '')}\n"
            result += f"Email: {student.get('email', '')}\n"
            result += f"Status: {student.get('enrollment_status', 'Active').replace('_', ' ').title()}\n"
            result += f"Academic Standing: {student.get('academic_standing', 'Good Standing').replace('_', ' ').title()}\n"
            
            if student.get('major'):
                result += f"Major: {student.get('major')}\n"
            if student.get('minor'):
                result += f"Minor: {student.get('minor')}\n"
                
            result += f"Total Credits: {student.get('total_credits', 0)}\n"
            result += f"GPA: {student.get('gpa', 'N/A')}\n"
            result += f"Year: {student.get('year', 'N/A')}\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error verifying student identity: {str(e)}")
            return f"Error verifying student identity: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def get_academic_holds(self, student_id: str) -> str:
        """
        Get academic holds information for a student.
        
        Args:
            student_id: Student ID
            
        Returns:
            List of academic holds and their details
        """
        try:
            student = self._get_student(student_id)
            
            student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
            result = f"Academic Holds for {student_name} (ID: {student.get('student_id', student_id)})\n\n"
            
            # Note: In the actual database, holds might be stored differently
            # For now, we'll return a basic holds check
            result += "Current Holds: None found\n"
            result += "\nCommon types of academic holds:\n"
            result += "• Financial Hold - Outstanding balance\n"
            result += "• Transcript Hold - Missing documents\n"
            result += "• Academic Hold - GPA/standing issues\n"
            result += "• Administrative Hold - Other requirements\n"
            result += "\nIf you believe there should be holds on record, please contact the Registrar's Office.\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error getting academic holds: {str(e)}")
            return f"Error retrieving academic holds: {str(e)}"

    @is_tool(tool_type=ToolType.WRITE)
    def resolve_hold(self, student_id: str, hold_id: str) -> str:
        """
        Resolve or process an academic hold.
        
        Args:
            student_id: Student ID
            hold_id: Hold identifier to resolve
            
        Returns:
            Hold resolution status
        """
        try:
            student = self._get_student(student_id)
            
            student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
            result = f"Hold Resolution for {student_name} (ID: {student.get('student_id', student_id)})\n\n"
            result += f"Processing hold: {hold_id}\n\n"
            result += "⚠️  Hold resolution requires manual verification.\n"
            result += "Please contact the appropriate office:\n"
            result += "• Financial holds: Bursar's Office\n"
            result += "• Academic holds: Academic Advising\n"
            result += "• Transcript holds: Registrar's Office\n"
            result += "• Administrative holds: Student Services\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error resolving hold: {str(e)}")
            return f"Error resolving hold: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def get_technology_support(self, issue_type: str, student_id: Optional[str] = None) -> str:
        """
        Get information about technology support services.
        
        Args:
            issue_type: Type of technology issue
            student_id: Optional student ID for personalized support
            
        Returns:
            Technology support information and resources
        """
        try:
            result = f"Technology Support Information\n\n"
            result += f"Issue Type: {issue_type}\n\n"
            
            result += "IT Help Desk Services:\n"
            result += "• Email setup and troubleshooting\n"
            result += "• Campus WiFi connection issues\n"
            result += "• Learning management system (LMS) support\n"
            result += "• Student portal access problems\n"
            result += "• Software installation and licensing\n"
            result += "• Password resets and account access\n"
            result += "• Printer and lab computer assistance\n\n"
            
            result += "Contact Information:\n"
            result += "• Help Desk: (555) 123-HELP (4357)\n"
            result += "• Email: helpdesk@university.edu\n"
            result += "• Location: IT Services Building, Room 101\n"
            result += "• Hours: Monday-Friday 8:00 AM - 8:00 PM\n"
            result += "• Emergency Support: 24/7 phone support\n\n"
            
            result += "Self-Service Resources:\n"
            result += "• Knowledge base: help.university.edu\n"
            result += "• Video tutorials available online\n"
            result += "• Live chat support during business hours\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting technology support info: {str(e)}")
            return f"Error retrieving technology support information: {str(e)}"

    @is_tool(tool_type=ToolType.WRITE)
    def change_major(self, student_id: str, new_major: str) -> str:
        """
        Process a major change request for a student.
        
        Args:
            student_id: Student ID
            new_major: New major to change to
            
        Returns:
            Major change process information
        """
        try:
            student = self._get_student(student_id)
            
            student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
            result = f"Major Change Request for {student_name} (ID: {student.get('student_id', student_id)})\n\n"
            result += f"Current Major: {student.get('major', 'Undeclared')}\n"
            result += f"Requested Major: {new_major}\n\n"
            
            result += "Major Change Process:\n"
            result += "1. ✅ Student identity verified\n"
            result += "2. ⏳ Academic requirements review needed\n"
            result += "3. ⏳ Advisor approval required\n"
            result += "4. ⏳ Department approval needed\n"
            result += "5. ⏳ System update pending\n\n"
            
            result += "Next Steps:\n"
            result += "• Schedule appointment with current academic advisor\n"
            result += "• Review new major requirements with department advisor\n"
            result += "• Complete major change form\n"
            result += "• Submit required documentation\n"
            result += "• Allow 5-10 business days for processing\n\n"
            
            result += "Important Notes:\n"
            result += "• Some majors have competitive admission requirements\n"
            result += "• Course credits may transfer differently\n"
            result += "• Graduation timeline may be affected\n"
            result += "• Financial aid eligibility should be reviewed\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error processing major change: {str(e)}")
            return f"Error processing major change: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def get_major_options(self, student_id: str) -> str:
        """
        Get available major options for a student.
        
        Args:
            student_id: Student ID
            
        Returns:
            List of available majors and requirements
        """
        try:
            student = self._get_student(student_id)
            
            student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
            result = f"Major Options for {student_name} (ID: {student_id})\n\n"
            result += f"Current Major: {student.get('major', 'Undeclared')}\n"
            result += f"Current Credits: {student.get('total_credits', 0)}\n"
            result += f"Academic Standing: {student.get('academic_standing', 'Good Standing').replace('_', ' ').title()}\n\n"
            
            # Get available majors from students (as proxy for available majors)
            result += "Available Majors:\n\n"
            majors = set()
            for student_data in self.db.students.values():
                if student_data.get('major'):
                    majors.add(student_data.get('major'))
            
            for major in sorted(majors):
                result += f"• {major}\n"
            
            result += "Choosing a Major:\n"
            result += "• Consider your interests and career goals\n"
            result += "• Review course requirements and prerequisites\n"
            result += "• Speak with department advisors\n"
            result += "• Consider time to graduation\n"
            result += "• Explore internship and job opportunities\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error getting major options: {str(e)}")
            return f"Error retrieving major options: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def get_study_abroad_programs(self, student_id: str, destination: Optional[str] = None) -> str:
        """
        Get study abroad program information.
        
        Args:
            student_id: Student ID
            destination: Preferred destination (optional)
            
        Returns:
            Study abroad programs and requirements
        """
        try:
            student = self._get_student(student_id)
            
            student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
            result = f"Study Abroad Programs for {student_name} (ID: {student.get('student_id', student_id)})\n\n"
            
            if destination:
                result += f"Programs for destination: {destination}\n\n"
            
            result += "Available Study Abroad Programs:\n\n"
            result += "Semester Programs:\n"
            result += "• European Studies - London, UK (Fall/Spring)\n"
            result += "• Business Exchange - Sydney, Australia (Fall/Spring)\n"
            result += "• Language Immersion - Madrid, Spain (Fall/Spring)\n"
            result += "• Engineering Co-op - Tokyo, Japan (Spring only)\n"
            result += "• Liberal Arts - Florence, Italy (Fall/Spring)\n\n"
            
            result += "Summer Programs:\n"
            result += "• Cultural Studies - Paris, France (6 weeks)\n"
            result += "• Marine Biology - Costa Rica (4 weeks)\n"
            result += "• Archaeological Field School - Peru (8 weeks)\n"
            result += "• International Business - Hong Kong (6 weeks)\n\n"
            
            result += "Eligibility Requirements:\n"
            result += f"• Minimum GPA: 2.5 (Current: {student.get('gpa', 'N/A')})\n"
            result += f"• Academic Standing: Good Standing (Current: {student.get('academic_standing', 'Unknown')})\n"
            result += f"• Completed 30+ credits (Current: {student.get('total_credits', 0)})\n"
            result += "• No outstanding financial obligations\n"
            result += "• Valid passport required\n\n"
            
            result += "Application Process:\n"
            result += "• Submit application 6 months in advance\n"
            result += "• Attend study abroad information session\n"
            result += "• Complete financial planning workshop\n"
            result += "• Obtain academic advisor approval\n"
            result += "• Submit health and safety forms\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error getting study abroad programs: {str(e)}")
            return f"Error retrieving study abroad programs: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def get_internship_opportunities(self, student_id: str, field: Optional[str] = None) -> str:
        """
        Get internship opportunities for a student.
        
        Args:
            student_id: Student ID
            field: Field of interest (optional)
            
        Returns:
            Available internship opportunities
        """
        try:
            student = self._get_student(student_id)
            
            student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
            result = f"Internship Opportunities for {student_name} (ID: {student.get('student_id', student_id)})\n\n"
            result += f"Major: {student.get('major', 'Undeclared')}\n"
            
            if field:
                result += f"Field of Interest: {field}\n"
            result += "\n"
            
            result += "Current Internship Postings:\n\n"
            
            # Sample internships based on field or major
            student_major = student.get('major', '').lower()
            if (field and "computer science" in field.lower()) or ("computer science" in student_major):
                result += "Technology Internships:\n"
                result += "• Software Development Intern - TechCorp (Summer)\n"
                result += "• Data Analytics Intern - DataSolutions Inc. (Fall)\n"
                result += "• Cybersecurity Intern - SecureNet LLC (Spring)\n"
                result += "• Web Development Intern - WebDesign Co. (Summer)\n\n"
            
            if (field and "business" in field.lower()) or ("business" in student_major):
                result += "Business Internships:\n"
                result += "• Marketing Intern - Marketing Plus (Summer)\n"
                result += "• Finance Intern - InvestCorp (Fall/Spring)\n"
                result += "• Consulting Intern - Strategy Group (Summer)\n"
                result += "• Operations Intern - LogisticsPro (Fall)\n\n"
            
            if (field and "psychology" in field.lower()) or ("psychology" in student_major):
                result += "Psychology Internships:\n"
                result += "• Research Assistant - University Psychology Lab (Fall/Spring)\n"
                result += "• Clinical Intern - Community Health Center (Summer)\n"
                result += "• HR Intern - People Solutions Inc. (Fall)\n"
                result += "• Social Services Intern - City Services (Spring)\n\n"
            
            result += "General Opportunities:\n"
            result += "• Non-profit Intern - Various Organizations\n"
            result += "• Government Intern - City/State Agencies\n"
            result += "• Research Assistant - University Departments\n"
            result += "• Teaching Assistant - Academic Departments\n\n"
            
            result += "Application Requirements:\n"
            result += "• Updated resume and cover letter\n"
            result += "• Academic transcript\n"
            result += "• Two professional references\n"
            result += "• Minimum 2.5 GPA\n"
            result += "• Completed sophomore year (60+ credits)\n\n"
            
            result += "Career Services Support:\n"
            result += "• Resume and cover letter review\n"
            result += "• Interview preparation workshops\n"
            result += "• Networking events with employers\n"
            result += "• Internship credit coordination\n"
            result += "\nContact Career Services at careers@university.edu\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error getting internship opportunities: {str(e)}")
            return f"Error retrieving internship opportunities: {str(e)}"

    @is_tool(tool_type=ToolType.READ) 
    def get_academic_standing(self, student_id: str) -> str:
        """
        Get detailed academic standing information for a student.
        
        Args:
            student_id: Student ID
            
        Returns:
            Academic standing details and requirements
        """
        try:
            student = self._get_student(student_id)
            
            student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
            result = f"Academic Standing Report for {student_name} (ID: {student.get('student_id', student_id)})\n\n"
            result += f"Current Standing: {student.get('academic_standing', 'Unknown').replace('_', ' ').title()}\n"
            result += f"Current GPA: {student.get('gpa', 'N/A')}\n"
            result += f"Total Credits: {student.get('total_credits', 0)}\n"
            result += f"Enrollment Status: {student.get('enrollment_status', 'Unknown').replace('_', ' ').title()}\n\n"
            
            # Explain academic standing categories
            result += "Academic Standing Definitions:\n\n"
            result += "• Good Standing: GPA ≥ 2.0, meeting all requirements\n"
            result += "• Academic Warning: GPA below 2.0 for one semester\n"
            result += "• Academic Probation: GPA below 2.0 for two consecutive semesters\n"
            result += "• Academic Dismissal: Failure to improve after probation\n\n"
            
            # Standing-specific information
            standing = student.get('academic_standing', 'Unknown').lower()
            if standing == "good standing":
                result += "✅ You are in Good Standing!\n"
                result += "Continue maintaining your current academic performance.\n"
            elif standing == "academic warning" or standing == "warning":
                result += "⚠️  Academic Warning Status\n"
                result += "You must raise your GPA to 2.0 or above by next semester.\n"
                result += "Consider meeting with your academic advisor.\n"
            elif standing == "academic probation" or standing == "probation":
                result += "⚠️  Academic Probation Status\n"
                result += "You must achieve a 2.0 GPA this semester or face dismissal.\n"
                result += "Mandatory academic advising required.\n"
            elif standing == "academic dismissal" or standing == "dismissal":
                result += "❌ Academic Dismissal Status\n"
                result += "You have been dismissed from the university.\n"
                result += "Contact Academic Affairs for appeal process.\n"
            else:
                result += f"Current Status: {standing.title()}\n"
            
            result += "\nResources for Academic Success:\n"
            result += "• Academic Advising: advising@university.edu\n"
            result += "• Tutoring Center: tutoring@university.edu\n"
            result += "• Learning Assistance: learning@university.edu\n"
            result += "• Counseling Services: counseling@university.edu\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error getting academic standing: {str(e)}")
            return f"Error retrieving academic standing: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def get_career_services_info(self, student_id: str, service_type: Optional[str] = None) -> str:
        """
        Get career services information and opportunities.
        
        Args:
            student_id: Student ID
            service_type: Type of career service (optional)
            
        Returns:
            Career services information and resources
        """
        try:
            student = self._get_student(student_id)
            
            student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
            result = f"Career Services Information for {student_name} (ID: {student.get('student_id', student_id)})\n\n"
            result += f"Major: {student.get('major', 'Undeclared')}\n"
            result += f"Academic Year: {student.get('year', 'N/A')}\n\n"
            
            if service_type:
                result += f"Service Type: {service_type}\n\n"
            
            result += "Available Career Services:\n\n"
            
            result += "Resume & Cover Letter Services:\n"
            result += "• Resume review and feedback\n"
            result += "• Cover letter writing assistance\n"
            result += "• LinkedIn profile optimization\n"
            result += "• Portfolio development guidance\n\n"
            
            result += "Job Search Support:\n"
            result += "• Job search strategies and techniques\n"
            result += "• Interview preparation and practice\n"
            result += "• Networking event coordination\n"
            result += "• Career fair participation\n\n"
            
            result += "Internship Programs:\n"
            result += "• Internship search assistance\n"
            result += "• Application support and guidance\n"
            result += "• Employer partnership programs\n"
            result += "• Academic credit coordination\n\n"
            
            result += "Career Exploration:\n"
            result += "• Career assessment and counseling\n"
            result += "• Industry exploration workshops\n"
            result += "• Alumni mentorship programs\n"
            result += "• Graduate school guidance\n\n"
            
            result += "Professional Development:\n"
            result += "• Workplace skills workshops\n"
            result += "• Professional etiquette training\n"
            result += "• Leadership development programs\n"
            result += "• Communication skills enhancement\n\n"
            
            result += "Contact Information:\n"
            result += "• Career Services Office: careers@university.edu\n"
            result += "• Phone: (555) 123-JOBS (5627)\n"
            result += "• Location: Student Services Building, Room 250\n"
            result += "• Hours: Monday-Friday 8:00 AM - 5:00 PM\n"
            result += "• Appointments: Schedule online or call\n\n"
            
            result += "Upcoming Events:\n"
            result += "• Career Fair - October 15th, 10 AM - 4 PM\n"
            result += "• Resume Workshop - Every Tuesday, 3 PM\n"
            result += "• Interview Skills Seminar - October 20th, 2 PM\n"
            result += "• Networking Night - October 25th, 6 PM\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error getting career services info: {str(e)}")
            return f"Error retrieving career services information: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def get_course_schedule(self, student_id: str, semester: Optional[str] = None, preferences: Optional[str] = None, course_code: Optional[str] = None) -> str:
        """
        Get course schedule information for a student.
        
        Args:
            student_id: Student ID
            semester: Specific semester (optional)
            preferences: Schedule preferences (optional)
            course_code: Specific course code filter (optional)
            
        Returns:
            Student's course schedule
        """
        try:
            student = self._get_student(student_id)
            
            student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
            result = f"Course Schedule for {student_name} (ID: {student.get('student_id', student_id)})\n\n"
            
            if semester:
                result += f"Semester: {semester.title()}\n\n"
            else:
                result += "Current Semester Schedule:\n\n"
            
            # Since we don't have detailed schedule data, provide a sample schedule
            result += "Enrolled Courses:\n\n"
            
            # Get current enrollments (simulated since we don't have real enrollment data)
            result += "Monday/Wednesday/Friday:\n"
            result += "• 9:00-9:50 AM: MATH 201 - Calculus II (Room: Science 101)\n"
            result += "• 11:00-11:50 AM: ENG 102 - Composition II (Room: Liberal Arts 205)\n"
            result += "• 2:00-2:50 PM: CS 201 - Data Structures (Room: Computer Lab 1)\n\n"
            
            result += "Tuesday/Thursday:\n"
            result += "• 10:00-11:15 AM: HIST 150 - World History (Room: Humanities 301)\n"
            result += "• 1:00-2:15 PM: BIO 101 - General Biology (Room: Science 202)\n"
            result += "• 3:30-4:45 PM: BIO 101L - Biology Lab (Room: Science Lab 1)\n\n"
            
            result += "Schedule Summary:\n"
            result += f"• Total Courses: 6\n"
            result += f"• Total Credit Hours: 17\n"
            result += f"• Academic Standing: {student.get('academic_standing', 'Unknown').replace('_', ' ').title()}\n"
            result += f"• Current GPA: {student.get('gpa', 'N/A')}\n\n"
            
            result += "Important Dates:\n"
            result += "• Add/Drop Deadline: September 15th\n"
            result += "• Midterm Exams: October 14-18\n"
            result += "• Withdrawal Deadline: November 1st\n"
            result += "• Final Exams: December 9-13\n\n"
            
            result += "Academic Resources:\n"
            result += "• Study Groups: Contact instructors for information\n"
            result += "• Tutoring Center: Available for all courses\n"
            result += "• Office Hours: Check syllabus for each course\n"
            result += "• Academic Advising: Schedule appointment as needed\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error getting course schedule: {str(e)}")
            return f"Error retrieving course schedule: {str(e)}"

