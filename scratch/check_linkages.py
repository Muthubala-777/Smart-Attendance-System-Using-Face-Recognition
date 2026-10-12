import re

with open('frontend/index.html', encoding='utf-8') as f:
    html = f.read()

with open('frontend/js/app.js', encoding='utf-8') as f:
    js = f.read()

required_ids = [
    'navStudentStartCamera', 'navStudentProfile', 'view-student-profile',
    'studentCameraSection', 'studentWebcamVideo', 'studentCameraOverlay',
    'studentCameraStandby', 'btnStudentStartCamera', 'btnStudentStopCamera',
    'btnScrollToCamera', 'studentProfileHeaderName', 'studentProfileHeaderRoll',
    'studentProfileHeaderDept', 'profilePagePhoto', 'profilePageName',
    'profilePageRoll', 'profilePageDept', 'profilePageYear', 'profilePageSection',
    'profilePageAcdYear', 'profilePageSemester', 'profilePageEmail', 'profilePagePhone',
    'profileStatTotal', 'profileStatPresent', 'profileStatAbsent', 'profileStatLate',
    'profileProgressPct', 'profileProgressBar', 'btnProfileGoToCamera', 'btnProfileGoToHistory'
]

all_ok = True
for rid in required_ids:
    in_html = f'id="{rid}"' in html or f"id='{rid}'" in html
    in_js = f'"{rid}"' in js or f"'{rid}'" in js
    print(f'{rid:28s} -> HTML: {in_html}, JS: {in_js}')
    if not in_html or not in_js:
        all_ok = False

print('\nALL LINKAGES OK?', all_ok)
