# Education Domain Standardized Data

## Student Data Standards

### Student IDs (8-digit format)
- EASY_1 persona: 10000001-10000020
- EASY_2 persona: 20000001-20000020  
- MEDIUM_1 persona: 30000001-30000020
- MEDIUM_2 persona: 40000001-40000020
- HARD_1 persona: 50000001-50000020

### Email Format
- Pattern: `firstname.lastname@university.edu`
- Examples: `sarah.chen@university.edu`, `alex.thompson@university.edu`

### Course Codes
- CS courses: CS101, CS102, CS201, CS202, CS301, CS302, CS401, CS402
- MATH courses: MATH101, MATH102, MATH201, MATH202, MATH301, MATH302
- ENG courses: ENG101, ENG102, ENG201, ENG202
- PSY courses: PSY101, PSY102, PSY201, PSY202, PSY301
- BUS courses: BUS101, BUS102, BUS201, BUS202, BUS301
- BIO courses: BIO101, BIO102, BIO201, BIO202
- CHEM courses: CHEM101, CHEM102, CHEM201, CHEM202
- PHYS courses: PHYS101, PHYS102, PHYS201, PHYS202

### Majors
- computer_science
- mathematics
- psychology
- business
- biology
- chemistry
- physics
- english
- undeclared

### Academic Standing
- good_standing
- warning
- probation
- suspension

### Semesters
- Fall2024
- Spring2025
- Summer2025

## Tool-Goal Alignment

### Course Management Goals
- course_registration → search_courses, register_for_course, check_prerequisites
- course_search → search_courses, get_course_details
- prerequisite_checking → check_prerequisites, get_course_details
- waitlist_management → add_to_waitlist, remove_from_waitlist

### Academic Planning Goals  
- major_exploration → get_major_options, get_degree_requirements
- major_change → change_major, get_degree_requirements
- graduation_planning → check_graduation_status, create_graduation_plan
- academic_advising → get_advisor_info, get_degree_requirements

### Financial Goals
- financial_aid → get_financial_aid_info, check_sap_status
- transcript_request → get_transcript, request_official_transcript

### Academic Status Goals
- academic_standing → get_academic_standing, get_academic_holds
- probation_planning → get_probation_requirements, get_academic_standing
- hold_resolution → resolve_hold, get_academic_holds

### Campus Services Goals
- campus_resources → search_campus_resources, get_library_info
- technology_support → get_technology_support
- career_guidance → get_career_services_info, get_internship_opportunities

### Study Abroad Goals
- study_abroad → get_study_abroad_programs, get_study_abroad_requirements
- credit_transfer → evaluate_transfer_credits, submit_transfer_credit_request
- internship_search → get_internship_opportunities

### Calendar Goals
- academic_calendar → get_academic_calendar, get_registration_dates

## Logical Goal Shift Patterns

### Soft Shifts (2 goals, natural progression)
- course_registration → major_exploration
- course_search → prerequisite_checking  
- financial_aid → academic_calendar
- academic_standing → probation_planning
- graduation_planning → transcript_request
- campus_resources → technology_support
- study_abroad → credit_transfer
- internship_search → career_guidance

### Hard Shifts (3 goals, complex pivots)
- course_registration → prerequisite_checking → waitlist_management
- major_exploration → academic_advising → career_guidance
- financial_aid → academic_standing → probation_planning
- graduation_planning → study_abroad → credit_transfer
- academic_standing → hold_resolution → course_registration
- campus_resources → technology_support → academic_calendar
