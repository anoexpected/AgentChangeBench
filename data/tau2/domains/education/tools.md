## Available Tools

Use exactly one tool at a time. For any write that changes enrollment, academic status, financial aid, or personal information, first show an action summary and obtain explicit "YES". All academic data follows university standards.

### Common types
- IDs: `student_id`, `course_id`, `enrollment_id`, `advisor_id`, `hold_id`, `aid_id`, `transcript_id`, `request_id`
- Academic levels: Freshman, Sophomore, Junior, Senior, Graduate
- Academic standing: good_standing, warning, probation, suspension
- All write tools accept `idempotency_key` and return an object with `status`

---

### Authentication and Student Lookup

**verify_student_identity(student_id, date_of_birth) → {verified: bool, auth_level: "L1"|"L2"}**  
Basic identity verification for student services.

**get_student_by_id(student_id) → Student**  
**get_student_by_email(email) → Student**  
**get_student_by_name_dob(full_name, date_of_birth) → Student**  
Use exactly one lookup path. After 3 failed attempts, propose transfer to registrar.

---

### Course Management

**search_courses(subject: str = None, level: str = None, instructor: str = None, semester: str = None) → [Course]**  
Search for courses by various criteria. Returns course details including prerequisites, schedule, capacity.

**get_course_details(course_id) → Course**  
Get detailed information about a specific course including syllabus, prerequisites, and enrollment status.

**check_prerequisites(course_id, student_id) → {eligible: bool, missing_prereqs: [str], recommendations: [str]}**  
Check if student meets course prerequisites and get recommendations for missing requirements.

**get_course_schedule(student_id, semester: str = None) → [Enrollment]**  
Get student's current or planned course schedule for a semester.

---

### Enrollment and Registration

**check_enrollment_status(student_id) → {status: str, holds: [Hold], can_register: bool}**  
Check student's enrollment status and any holds preventing registration.

**register_for_course(student_id, course_id, semester) → {enrollment_id, status: "Registered"|"Waitlisted"|"Error"}**  
Register student for a course. Requires L2 and "YES" if course is full or has restrictions.

**drop_course(enrollment_id, reason: str = None) → {status}**  
Drop a course enrollment. Requires L2 and "YES" if past drop deadline.

**add_to_waitlist(student_id, course_id, semester) → {waitlist_position, status}**  
Add student to course waitlist.

**remove_from_waitlist(student_id, course_id) → {status}**  
Remove student from course waitlist.

---

### Academic Records and Transcripts

**get_transcript(student_id, official: bool = False) → Transcript**  
Get student's academic transcript. Official transcripts require L2 authentication.

**request_official_transcript(student_id, delivery_method: str, recipient_address: str = None) → {request_id, status, delivery_eta}**  
Request official transcript delivery. Requires L2 and "YES".

**get_grade_report(student_id, semester: str = None) → [Grade]**  
Get grade report for current or specific semester.

**calculate_gpa(student_id, semester: str = None, cumulative: bool = True) → {gpa: float, credits_attempted: int, credits_earned: int}**  
Calculate student's GPA for specific semester or cumulative.

---

### Academic Advising and Planning

**get_advisor_info(student_id) → Advisor**  
Get student's assigned academic advisor information and contact details.

**get_degree_requirements(major: str, student_id: str = None) → DegreeRequirements**  
Get degree requirements for a major. If student_id provided, shows progress toward completion.

**check_graduation_status(student_id) → {eligible: bool, missing_requirements: [str], estimated_graduation: str}**  
Check if student is eligible for graduation and identify missing requirements.

**create_graduation_plan(student_id, target_graduation_date: str = None) → {plan_id, status, recommendations: [str]}**  
Create or update graduation plan for student.

**get_major_options() → [Major]**  
Get list of available majors and their basic requirements.

**change_major(student_id, new_major: str, reason: str = None) → {status, effective_date}**  
Change student's major. Requires L2 and "YES".

---

### Financial Aid and Billing

**get_financial_aid_info(student_id) → FinancialAid**  
Get student's financial aid package, status, and requirements.

**check_sap_status(student_id) → {status: str, completion_rate: float, gpa: float, max_timeframe: str}**  
Check Satisfactory Academic Progress (SAP) status for financial aid eligibility.

**submit_sap_appeal(student_id, reason: str, supporting_docs: [str] = None) → {appeal_id, status, review_eta}**  
Submit SAP appeal for financial aid reinstatement. Requires L2 and "YES".

**get_billing_info(student_id, semester: str = None) → BillingInfo**  
Get student's billing information including charges, payments, and balance.

**get_payment_plan_options(student_id) → [PaymentPlan]**  
Get available payment plan options for outstanding balance.

---

### Academic Standing and Holds

**get_academic_standing(student_id) → {standing: str, gpa: float, credits_attempted: int, status_date: str}**  
Get student's current academic standing (good, warning, probation, suspension).

**get_academic_holds(student_id) → [Hold]**  
Get list of academic holds preventing registration or other services.

**resolve_hold(hold_id, resolution_method: str, documentation: [str] = None) → {status, resolution_date}**  
Resolve an academic hold. Requires L2 and "YES".

**get_probation_requirements(student_id) → {requirements: [str], deadline: str, support_resources: [str]}**  
Get requirements for getting off academic probation.

---

### Campus Resources and Services

**search_campus_resources(resource_type: str = None, location: str = None, keyword: str = None) → [CampusResource]**  
Search for campus resources and services.

**get_library_info() → LibraryInfo**  
Get library hours, services, and study space availability.

**get_technology_support(issue_type: str = None) → [SupportOption]**  
Get technology support options and contact information.

**get_career_services_info() → CareerServices**  
Get career services information including counseling, job board, and events.

**get_study_abroad_programs(country: str = None, program_type: str = None) → [StudyAbroadProgram]**  
Get available study abroad programs and requirements.

**get_internship_opportunities(major: str = None, location: str = None) → [Internship]**  
Get available internship opportunities and application requirements.

---

### Academic Calendar and Scheduling

**get_academic_calendar(semester: str = None, event_type: str = None) → [CalendarEvent]**  
Get academic calendar information including important dates and deadlines.

**get_registration_dates(student_id, semester: str = None) → {registration_opens: str, registration_closes: str, priority_level: str}**  
Get student's registration dates and priority level.

**get_exam_schedule(student_id, semester: str = None) → [Exam]**  
Get student's exam schedule for current or specific semester.

---

### Study Abroad and Credit Transfer

**get_study_abroad_requirements(program_id: str = None) → StudyAbroadRequirements**  
Get requirements for study abroad programs.

**submit_study_abroad_application(program_id: str, student_id: str, documents: [str]) → {application_id, status, review_eta}**  
Submit study abroad application. Requires L2 and "YES".

**evaluate_transfer_credits(student_id, institution: str, courses: [str]) → {evaluated_credits: [TransferCredit], total_credits: int}**  
Evaluate transfer credits from another institution.

**submit_transfer_credit_request(student_id, course_details: [dict]) → {request_id, status, review_eta}**  
Submit request for transfer credit evaluation. Requires L2 and "YES".

---

### Profile and Contact Information

**get_student_profile(student_id) → StudentProfile**  
Get student's profile information including contact details and preferences.

**update_contact_info(student_id, phone: str = None, email: str = None, address: dict = None) → {status}**  
Update student's contact information. Requires L2 and "YES".

**update_emergency_contact(student_id, contact_info: dict) → {status}**  
Update emergency contact information. Requires L2 and "YES".

---

### Errors and retries

Every tool returns success or `{error_code, error_message, retriable: bool}`.  
On retriable true, offer one retry. After two failures in a row, propose a human transfer.

---

### Re-auth triggers

Re-auth is required when any of these are true:  
- First enrollment change in session, or switching from inquiry to any write operation  
- Major change, hold resolution, financial aid appeal, transcript request  
- Course registration, course drop, study abroad application  
- Contact information update, emergency contact update  
- Any operation requiring "YES" confirmation, or step-up timeout older than 5 minutes
