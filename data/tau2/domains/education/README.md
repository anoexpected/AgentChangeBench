# Education Domain

This domain simulates academic advising and student services interactions in a university setting.

## Overview

The education domain provides realistic scenarios for evaluating conversational AI agents in academic contexts, including course registration, academic advising, financial aid, and student services.

## Task Categories

### Course Planning (25 tasks)
- Course search and registration
- Prerequisite checking
- Schedule optimization
- Waitlist management

### Academic Advising (25 tasks)
- Major exploration and changes
- Graduation planning
- Degree requirements
- Career guidance

### Financial Aid (20 tasks)
- Aid package inquiries
- SAP status checks
- Appeal processes
- Payment planning

### Student Services (15 tasks)
- Transcript requests
- Academic holds
- Technology support
- Campus resources

### Campus Resources (15 tasks)
- Library services
- Study spaces
- Computer labs
- Career services

## Goal-Shift Patterns

### Soft Shifts (50 tasks)
Natural progression between related goals:
- Course registration → Major exploration
- Financial aid → Academic planning
- Transcript request → Graduation planning

### Hard Shifts (50 tasks)
Complex pivots requiring adaptation:
- Course planning → Academic probation → Recovery planning
- Financial aid → Major change → Career guidance
- Registration → Study abroad → Credit transfer

## Personas

### EASY_1: Polite, Detail-Oriented
- Asks clarifying questions
- Appreciates explanations
- Patient with processes

### EASY_2: Confused, Panicky
- Easily overwhelmed
- Needs reassurance
- Sudden goal changes

### MEDIUM_1: Impatient, Business-Focused
- Time-conscious
- Results-oriented
- Quick decisions

### MEDIUM_2: Curious, Learning
- Asks many questions
- Enthusiastic
- Exploration-driven

### HARD_1: Suspicious, Demanding
- Questions everything
- High expectations
- Confrontational

## Tools Available

- `search_courses`: Find courses by subject, level, instructor
- `check_prerequisites`: Verify course requirements
- `get_degree_requirements`: Get major requirements
- `check_enrollment_status`: Check student status
- `get_transcript`: Access academic records
- `get_financial_aid_info`: Get aid information
- `get_academic_calendar`: Get calendar information
- `search_campus_resources`: Find campus services
- `get_advisor_info`: Get advisor information
- `check_graduation_status`: Check graduation eligibility

## Evaluation Criteria

### Task Success Rate (TSR)
- Course registration success
- Accurate information provided
- Appropriate guidance given
- Error handling quality

### Goal Shift Recovery Time (GSRT)
- Time to adapt to new goals
- Smoothness of transitions
- Context retention
- Proactive assistance

### Communication Quality
- Clarity of explanations
- Appropriate tone
- Persona consistency
- Professional demeanor

### Action Execution
- Tool usage efficiency
- Accurate data retrieval
- Appropriate recommendations
- Problem resolution

## Sample Scenarios

### Scenario 1: Course Registration → Major Change
**Student**: "I want to register for my spring courses, but I'm thinking about switching to Computer Science."
**Goals**: course_registration → major_exploration
**Tools**: search_courses, get_degree_requirements, check_prerequisites

### Scenario 2: Financial Aid → Academic Probation
**Student**: "I need to check my financial aid, but I just found out I'm on academic probation!"
**Goals**: financial_aid → academic_standing → probation_planning
**Tools**: get_financial_aid_info, get_transcript, get_advisor_info

### Scenario 3: Graduation Planning → Study Abroad
**Student**: "I'm checking my graduation requirements, but I want to study abroad next semester."
**Goals**: graduation_planning → study_abroad → credit_transfer
**Tools**: check_graduation_status, search_campus_resources, get_advisor_info

## Integration Notes

This domain integrates with the AgentChangeBench framework and follows the same evaluation patterns as banking, airline, and retail domains. The education domain provides unique challenges in academic policy complexity, multi-step planning processes, and diverse student needs.
