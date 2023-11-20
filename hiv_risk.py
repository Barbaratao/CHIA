def assess_hiv_risk():
    questions = {
        'sex_with_men': "Have you had unprotected sexual intercourse with men in the past 3 months? (Yes/No): ",
        'multiple_partners': "Have you had multiple sexual partners in the past 12 months? (Yes/No): ",
        'iv_drug_use': "Have you used intravenous drugs or shared needles? (Yes/No): ",
        'partner_hiv_positive/unknown': "Do you have a sexual partner who is HIV positive/ has unknown HIV status? (Yes/No): ",
        'std_history': "Have you been diagnosed with a sexually transmitted disease (STD) in the past 12 months? (Yes/No): ",
        # Add more questions as necessary
    }

    high_risk = False
    responses = {}

    print("HIV Risk Assessment Questionnaire\n")

    for key, question in questions.items():
        response = input(question).strip().lower()
        responses[key] = response
        if response == 'yes':
            high_risk = True

    if high_risk:
        print("\nBased on your responses, you may be at a higher risk for HIV. It is recommended to consider taking PrEP to protect from HIV infection.")
    else:
        print("\nBased on your responses, your risk for HIV appears to be lower. However, continue to practice safe behaviors and consult a healthcare professional for personalized advice.")

    return responses
