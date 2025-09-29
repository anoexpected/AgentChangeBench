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
        level: Optional[int] = None,
        instructor: Optional[str] = None,
        keyword: Optional[str] = None,
    ) -> str:
        """
        Find courses by subject, level, instructor, or keyword.
        
        Args:
            subject: Course subject (e.g., 'CS', 'MATH')
            level: Course level (e.g., 100, 200, 300)
            instructor: Instructor name
            keyword: Search keyword in course title or description
            
        Returns:
            List of matching courses with details
        """
        try:
            matching_courses = []
            
            for course_id, course in self.db.courses.items():
                # Apply filters
                if subject and course.get("subject", "").upper() != subject.upper():
                    continue
                if level and course.get("level") != level:
                    continue
                if instructor and instructor.lower() not in (course.get("instructor", "") or "").lower():
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
                    "subject": course.get("subject", course.get("department", "")),
                    "level": course.get("level", ""),
                    "credits": course.get("credits", ""),
                    "instructor": course.get("instructor", "TBA"),
                    "status": course.get("status", "Available"),
                    "enrollment": f"{course.get('current_enrollment', 0)}/{course.get('max_enrollment', 'N/A')}",
                    "waitlist": course.get("waitlist_count", 0),
                    "prerequisites": course.get("prerequisites", []),
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
    def check_prerequisites(self, course_id: str, student_id: Optional[str] = None) -> str:
        """
        Verify course requirements and check if student has met prerequisites.
        
        Args:
            course_id: Course ID to check prerequisites for
            student_id: Student ID to check eligibility (optional)
            
        Returns:
            Prerequisites information and eligibility status
        """
        try:
            course = self._get_course(course_id)
            
            result = f"Prerequisites for {course.course_id}: {course.title}\n\n"
            
            if not course.prerequisites and not course.corequisites:
                result += "No prerequisites required for this course.\n"
            else:
                if course.prerequisites:
                    result += f"Prerequisites: {', '.join(course.prerequisites)}\n"
                if course.corequisites:
                    result += f"Corequisites: {', '.join(course.corequisites)}\n"
            
            # Check student eligibility if student_id provided
            if student_id:
                try:
                    student = self._get_student(student_id)
                    student_enrollments = [e for e in self.db.enrollments if e.student_id == student_id]
                    completed_courses = [e.course_id for e in student_enrollments if e.grade and e.grade not in ['F', 'W']]
                    
                    # Check prerequisites
                    missing_prereqs = []
                    for prereq in course.prerequisites:
                        if prereq not in completed_courses:
                            missing_prereqs.append(prereq)
                    
                    if missing_prereqs:
                        result += f"\n❌ Missing prerequisites: {', '.join(missing_prereqs)}\n"
                        result += "You must complete these courses before enrolling.\n"
                    else:
                        result += "\n✅ All prerequisites satisfied!\n"
                        
                        # Check if course is full
                        if course.status == CourseStatus.FULL:
                            result += "⚠️  Course is currently full. You may join the waitlist.\n"
                        elif course.status == CourseStatus.AVAILABLE:
                            result += "Course is available for enrollment.\n"
                        
                except ValueError as ve:
                    result += f"\n⚠️  Could not verify eligibility: {str(ve)}\n"
            
            return result
            
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error checking prerequisites: {str(e)}")
            return f"Error checking prerequisites: {str(e)}"

    @is_tool(tool_type=ToolType.READ)
    def get_degree_requirements(self, major: str) -> str:
        """
        Get major requirements and degree information.
        
        Args:
            major: Major name or ID
            
        Returns:
            Detailed degree requirements information
        """
        try:
            major_obj = self._get_major(major)
            
            result = f"Degree Requirements for {major_obj.name}\n"
            result += f"Department: {major_obj.department}\n\n"
            result += f"Total Credits Required: {major_obj.required_credits}\n"
            result += f"Minimum GPA: {major_obj.minimum_gpa}\n"
            result += f"Elective Credits: {major_obj.elective_credits}\n\n"
            
            if major_obj.required_courses:
                result += "Required Courses:\n"
                for course_id in major_obj.required_courses:
                    try:
                        course = self._get_course(course_id)
                        result += f"• {course_id}: {course.title} ({course.credits} credits)\n"
                    except ValueError:
                        result += f"• {course_id}: Course details not available\n"
            
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
    def check_enrollment_status(self, student_id: str) -> str:
        """
        Check student enrollment status and information.
        
        Args:
            student_id: Student ID
            
        Returns:
            Student enrollment status and academic information
        """
        try:
            student = self._get_student(student_id)
            
            result = f"Enrollment Status for {student.name} (ID: {student.student_id})\n\n"
            result += f"Status: {student.status.value.replace('_', ' ').title()}\n"
            result += f"Academic Standing: {student.academic_standing.value.replace('_', ' ').title()}\n"
            result += f"Major: {student.major or 'Undeclared'}\n"
            if student.minor:
                result += f"Minor: {student.minor}\n"
            result += f"GPA: {student.gpa or 'N/A'}\n"
            result += f"Total Credits: {student.total_credits}\n"
            result += f"Enrollment Date: {student.enrollment_date}\n"
            
            # Get current enrollments
            current_enrollments = [
                e for e in self.db.enrollments 
                if e.student_id == student_id and not e.grade
            ]
            
            if current_enrollments:
                result += f"\nCurrent Enrollments ({len(current_enrollments)} courses):\n"
                total_credits = 0
                for enrollment in current_enrollments:
                    try:
                        course = self._get_course(enrollment.course_id)
                        result += f"• {enrollment.course_id}: {course.title} ({course.credits} credits)\n"
                        total_credits += course.credits
                    except ValueError:
                        result += f"• {enrollment.course_id}: Course details not available\n"
                result += f"Total Credits This Semester: {total_credits}\n"
            else:
                result += "\nNo current enrollments found.\n"
            
            # Check for advisor
            if student.advisor_id:
                try:
                    advisor = self._get_advisor(student.advisor_id)
                    result += f"\nAcademic Advisor: {advisor.name}\n"
                    result += f"Email: {advisor.email}\n"
                    result += f"Office: {advisor.office_location}\n"
                    result += f"Office Hours: {advisor.office_hours}\n"
                except ValueError:
                    result += f"\nAdvisor ID: {student.advisor_id} (details not available)\n"
            
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
            student_enrollments = [e for e in self.db.enrollments if e.student_id == student_id]
            
            result = f"Official Transcript for {student.name} (ID: {student.student_id})\n"
            result += f"Email: {student.email}\n"
            result += f"Major: {student.major or 'Undeclared'}\n"
            result += f"Academic Standing: {student.academic_standing.value.replace('_', ' ').title()}\n\n"
            
            # Group by semester and year
            enrollments_by_term = {}
            for enrollment in student_enrollments:
                term_key = f"{enrollment.semester.value.title()} {enrollment.year}"
                if term_key not in enrollments_by_term:
                    enrollments_by_term[term_key] = []
                enrollments_by_term[term_key].append(enrollment)
            
            total_credits_earned = 0
            total_grade_points = 0
            total_attempted = 0
            
            # Grade point mapping
            grade_points = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}
            
            for term in sorted(enrollments_by_term.keys()):
                result += f"\n{term}:\n"
                term_credits = 0
                term_grade_points = 0
                
                for enrollment in enrollments_by_term[term]:
                    try:
                        course = self._get_course(enrollment.course_id)
                        grade_display = enrollment.grade or "In Progress"
                        result += f"  {enrollment.course_id}: {course.title} - {course.credits} credits - {grade_display}\n"
                        
                        if enrollment.grade and enrollment.grade in grade_points:
                            term_credits += course.credits
                            term_grade_points += grade_points[enrollment.grade] * course.credits
                            total_attempted += course.credits
                            if enrollment.grade != 'F':
                                total_credits_earned += course.credits
                    except ValueError:
                        result += f"  {enrollment.course_id}: Course details not available - {enrollment.grade or 'In Progress'}\n"
                
                if term_credits > 0:
                    term_gpa = term_grade_points / term_credits
                    result += f"  Term GPA: {term_gpa:.2f}, Credits: {term_credits}\n"
                    total_grade_points += term_grade_points
            
            # Calculate overall GPA
            overall_gpa = total_grade_points / total_attempted if total_attempted > 0 else 0.0
            
            result += f"\n--- Summary ---\n"
            result += f"Total Credits Earned: {total_credits_earned}\n"
            result += f"Total Credits Attempted: {total_attempted}\n"
            result += f"Overall GPA: {overall_gpa:.2f}\n"
            result += f"Academic Standing: {student.academic_standing.value.replace('_', ' ').title()}\n"
            
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
    def search_campus_resources(self, resource_type: Optional[str] = None, keyword: Optional[str] = None) -> str:
        """
        Find campus services and resources.
        
        Args:
            resource_type: Type of resource (library, lab, service)
            keyword: Search keyword
            
        Returns:
            Available campus resources information
        """
        try:
            resources = self.db.campus_resources
            
            # Apply filters
            if resource_type:
                resources = [r for r in resources if resource_type.lower() in r.type.lower()]
            
            if keyword:
                resources = [
                    r for r in resources 
                    if keyword.lower() in r.name.lower() or 
                       keyword.lower() in r.description.lower() or
                       any(keyword.lower() in service.lower() for service in r.services)
                ]
            
            if not resources:
                return "No campus resources found matching your criteria."
            
            result = f"Campus Resources ({len(resources)} found):\n\n"
            
            # Group by type
            resources_by_type = {}
            for resource in resources:
                if resource.type not in resources_by_type:
                    resources_by_type[resource.type] = []
                resources_by_type[resource.type].append(resource)
            
            for res_type in sorted(resources_by_type.keys()):
                result += f"{res_type.title()}:\n"
                
                for resource in resources_by_type[res_type]:
                    result += f"\n• {resource.name}\n"
                    result += f"  Location: {resource.location}\n"
                    result += f"  Hours: {resource.hours}\n"
                    result += f"  Contact: {resource.contact_info}\n"
                    result += f"  Description: {resource.description}\n"
                    
                    if resource.services:
                        result += f"  Services: {', '.join(resource.services)}\n"
                
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
                if not student.advisor_id:
                    return f"No advisor assigned to student {student_id}. Please contact the academic advising office."
                advisor_id = student.advisor_id
            
            if not advisor_id:
                return "Please provide either a student ID or advisor ID."
            
            advisor = self._get_advisor(advisor_id)
            
            result = f"Academic Advisor Information\n\n"
            result += f"Name: {advisor.name}\n"
            result += f"Department: {advisor.department}\n"
            result += f"Email: {advisor.email}\n"
            result += f"Office Location: {advisor.office_location}\n"
            result += f"Office Hours: {advisor.office_hours}\n"
            
            if advisor.specializations:
                result += f"Specializations: {', '.join(advisor.specializations)}\n"
            
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
            
            result = f"Graduation Status for {student.name} (ID: {student.student_id})\n\n"
            
            # Basic requirements check
            min_credits = 120
            min_gpa = 2.0
            min_major_gpa = 2.5
            
            result += f"Progress Summary:\n"
            result += f"• Total Credits: {student.total_credits}/{min_credits} "
            result += "✅" if student.total_credits >= min_credits else "❌"
            result += "\n"
            
            result += f"• Overall GPA: {student.gpa or 'N/A'}/{min_gpa} "
            if student.gpa:
                result += "✅" if student.gpa >= min_gpa else "❌"
            else:
                result += "❌ (GPA not calculated)"
            result += "\n"
            
            result += f"• Academic Standing: {student.academic_standing.value.replace('_', ' ').title()} "
            result += "✅" if student.academic_standing == AcademicStanding.GOOD_STANDING else "❌"
            result += "\n"
            
            # Major requirements check
            if student.major:
                try:
                    major = self._get_major(student.major)
                    result += f"\nMajor Requirements ({major.name}):\n"
                    
                    # Get student's enrollments in major courses
                    student_enrollments = [e for e in self.db.enrollments if e.student_id == student_id]
                    completed_major_courses = []
                    major_credits = 0
                    major_grade_points = 0
                    
                    grade_points = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}
                    
                    for enrollment in student_enrollments:
                        if enrollment.course_id in major.required_courses and enrollment.grade and enrollment.grade != 'F':
                            completed_major_courses.append(enrollment.course_id)
                            try:
                                course = self._get_course(enrollment.course_id)
                                major_credits += course.credits
                                if enrollment.grade in grade_points:
                                    major_grade_points += grade_points[enrollment.grade] * course.credits
                            except ValueError:
                                pass
                    
                    remaining_major_courses = [c for c in major.required_courses if c not in completed_major_courses]
                    
                    result += f"• Required Courses: {len(completed_major_courses)}/{len(major.required_courses)} completed\n"
                    if remaining_major_courses:
                        result += f"  Remaining: {', '.join(remaining_major_courses)}\n"
                    
                    result += f"• Major Credits: {major_credits}/{major.required_credits} "
                    result += "✅" if major_credits >= major.required_credits else "❌"
                    result += "\n"
                    
                    major_gpa = major_grade_points / major_credits if major_credits > 0 else 0.0
                    result += f"• Major GPA: {major_gpa:.2f}/{min_major_gpa} "
                    result += "✅" if major_gpa >= min_major_gpa else "❌"
                    result += "\n"
                    
                except ValueError:
                    result += f"\nMajor '{student.major}' details not available.\n"
            else:
                result += "\n❌ No major declared. You must declare a major to graduate.\n"
            
            # General education requirements (simplified)
            result += f"\nGeneral Education Requirements:\n"
            result += f"• Estimated: 40-45 credits required\n"
            result += f"• Residency: 30 credits at university required\n"
            
            # Overall graduation eligibility
            eligible = (
                student.total_credits >= min_credits and
                student.gpa and student.gpa >= min_gpa and
                student.academic_standing == AcademicStanding.GOOD_STANDING and
                student.major
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
            
            result = f"Academic Holds for {student.name} (ID: {student.student_id})\n\n"
            
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
            
            result = f"Hold Resolution for {student.name} (ID: {student.student_id})\n\n"
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
    def get_technology_support(self, issue_type: str) -> str:
        """
        Get information about technology support services.
        
        Args:
            issue_type: Type of technology issue
            
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
            
            result = f"Major Change Request for {student.name} (ID: {student.student_id})\n\n"
            result += f"Current Major: {student.major or 'Undeclared'}\n"
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
            
            result = f"Study Abroad Programs for {student.name} (ID: {student.student_id})\n\n"
            
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
            result += f"• Minimum GPA: 2.5 (Current: {student.gpa or 'N/A'})\n"
            result += f"• Academic Standing: Good Standing (Current: {student.academic_standing.value})\n"
            result += "• Completed 30+ credits (Current: {student.total_credits})\n"
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
            
            result = f"Internship Opportunities for {student.name} (ID: {student.student_id})\n\n"
            result += f"Major: {student.major or 'Undeclared'}\n"
            
            if field:
                result += f"Field of Interest: {field}\n"
            result += "\n"
            
            result += "Current Internship Postings:\n\n"
            
            # Sample internships based on field or major
            if field and "computer science" in field.lower() or (student.major and "computer science" in student.major.lower()):
                result += "Technology Internships:\n"
                result += "• Software Development Intern - TechCorp (Summer)\n"
                result += "• Data Analytics Intern - DataSolutions Inc. (Fall)\n"
                result += "• Cybersecurity Intern - SecureNet LLC (Spring)\n"
                result += "• Web Development Intern - WebDesign Co. (Summer)\n\n"
            
            if field and "business" in field.lower() or (student.major and "business" in student.major.lower()):
                result += "Business Internships:\n"
                result += "• Marketing Intern - Marketing Plus (Summer)\n"
                result += "• Finance Intern - InvestCorp (Fall/Spring)\n"
                result += "• Consulting Intern - Strategy Group (Summer)\n"
                result += "• Operations Intern - LogisticsPro (Fall)\n\n"
            
            if field and "psychology" in field.lower() or (student.major and "psychology" in student.major.lower()):
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
            
            result = f"Academic Standing Report for {student.name} (ID: {student.student_id})\n\n"
            result += f"Current Standing: {student.academic_standing.value.replace('_', ' ').title()}\n"
            result += f"Current GPA: {student.gpa or 'N/A'}\n"
            result += f"Total Credits: {student.total_credits}\n"
            result += f"Enrollment Status: {student.status.value.replace('_', ' ').title()}\n\n"
            
            # Explain academic standing categories
            result += "Academic Standing Definitions:\n\n"
            result += "• Good Standing: GPA ≥ 2.0, meeting all requirements\n"
            result += "• Academic Warning: GPA below 2.0 for one semester\n"
            result += "• Academic Probation: GPA below 2.0 for two consecutive semesters\n"
            result += "• Academic Dismissal: Failure to improve after probation\n\n"
            
            # Standing-specific information
            if student.academic_standing == AcademicStanding.GOOD_STANDING:
                result += "✅ You are in Good Standing!\n"
                result += "Continue maintaining your current academic performance.\n"
            elif student.academic_standing == AcademicStanding.ACADEMIC_WARNING:
                result += "⚠️  Academic Warning Status\n"
                result += "You must raise your GPA to 2.0 or above by next semester.\n"
                result += "Consider meeting with your academic advisor.\n"
            elif student.academic_standing == AcademicStanding.ACADEMIC_PROBATION:
                result += "⚠️  Academic Probation Status\n"
                result += "You must achieve a 2.0 GPA this semester or face dismissal.\n"
                result += "Mandatory academic advising required.\n"
            elif student.academic_standing == AcademicStanding.ACADEMIC_DISMISSAL:
                result += "❌ Academic Dismissal Status\n"
                result += "You have been dismissed from the university.\n"
                result += "Contact Academic Affairs for appeal process.\n"
            
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

