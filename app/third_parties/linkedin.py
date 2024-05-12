import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()


def scrape_linkeding_profile(profile_url: str , mock = False):
    """Scrape linkedin profile information,
    manually scrape information from linkedin profile"""

    api_key =  os.getenv("PROXYCURL_API_KEY")
    headers = {'Authorization': 'Bearer ' + api_key}
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    params = {
        'linkedin_profile_url': 'https://linkedin.com/in/pericles-buarque/',
        'extra': 'include',
        'github_profile_id': 'include',
        'facebook_profile_id': 'include',
        'twitter_profile_id': 'include',
        'personal_contact_number': 'include',
        'personal_email': 'include',
        'inferred_salary': 'include',
        'skills': 'include',
        'use_cache': 'if-present',
        'fallback_to_cache': 'on-error',
    }

    response = requests.get(api_endpoint,
                            params=params,
                            headers=headers)
    data = response.json()
    # data = {
    #     k: v
    #     for k, v in data.items()
    #     if v not in ([], "", "", None)
    #     and k not in ["people_also_viewed", "certifications"]
    # }
    # if data.get("groups"):
    #     for group_dict in data.get("groups"):
    #         group_dict.pop("profile_pic_url")

    return data

# if __name__ == '__main__':

#     data = scrape_linkeding_profile(profile_url="https://linkedin.com/in/pericles-buarque/", mock = False)
#     print(data)

#     with open('profile_data.json', 'w') as file:
#         json.dump(data, file, indent=4)



