#!/usr/bin/env python3
"""Test the loan approval API with sample applications."""
import requests
import json

API_URL = "http://localhost:8000"

def test_application(profile_name, application_data):
    """Submit an application and display the results."""
    print("=" * 70)
    print(f"TEST: {profile_name}")
    print("=" * 70)

    response = requests.post(f"{API_URL}/predict", json=application_data)

    if response.status_code != 200:
        print(f"❌ Error: {response.json()}")
        return

    data = response.json()

    print(f"\n✅ DECISION: {data['decision']}")
    print(f"📊 Confidence: {data['confidence']:.1f}%")
    print(f"📈 Approval Probability: {data['probability_approved']:.1f}%")
    print(f"📉 Rejection Probability: {data['probability_rejected']:.1f}%")

    print(f"\n💡 REASONING:")
    print(f"{data['reasoning']}")

    print(f"\n🟢 TOP POSITIVE FACTORS (helping approval):")
    for i, factor in enumerate(data['top_positive_factors'][:3], 1):
        print(f"   {i}. {factor['feature']:<22} = {factor['value']:>8.2f}  →  +{factor['contribution']:.4f}")

    print(f"\n🔴 TOP NEGATIVE FACTORS (hurting approval):")
    for i, factor in enumerate(data['top_negative_factors'][:3], 1):
        print(f"   {i}. {factor['feature']:<22} = {factor['value']:>8.2f}  →  {factor['contribution']:>.4f}")

    print(f"\n📝 Application ID: #{data['application_id']}\n")


if __name__ == "__main__":
    # Test 1: High approval probability
    test_application(
        "HIGH APPROVAL - Graduate, Married, Good Credit, High Income",
        {
            "gender": "Male",
            "married": "Yes",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": 8000,
            "coapplicant_income": 2000,
            "loan_amount": 120,
            "loan_amount_term": 360,
            "credit_history": 1.0,
            "property_area": "Urban"
        }
    )

    # Test 2: Low approval probability
    test_application(
        "LOW APPROVAL - Not Graduate, Single, Bad Credit, Low Income",
        {
            "gender": "Female",
            "married": "No",
            "dependents": "3+",
            "education": "Not Graduate",
            "self_employed": "Yes",
            "applicant_income": 2500,
            "coapplicant_income": 0,
            "loan_amount": 250,
            "loan_amount_term": 120,
            "credit_history": 0.0,
            "property_area": "Rural"
        }
    )

    # Test 3: Edge case
    test_application(
        "EDGE CASE - High Loan Amount with Moderate Income",
        {
            "gender": "Male",
            "married": "Yes",
            "dependents": "2",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": 5000,
            "coapplicant_income": 1500,
            "loan_amount": 200,
            "loan_amount_term": 360,
            "credit_history": 1.0,
            "property_area": "Semiurban"
        }
    )
