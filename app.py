import os
import requests

from flask import Flask, request
from github import Github, GithubIntegration


app = Flask(__name__)
# MAKE SURE TO CHANGE TO YOUR APP NUMBER!!!!!
app_id = '<Your_App_Number_here>'
# Read the bot certificate
with open(
        os.path.normpath(os.path.expanduser('~/.certs/github/bot_key.pem')),
        'r'
) as cert_file:
    app_key = cert_file.read()

# Create an GitHub integration instance
git_integration = GithubIntegration(
    app_id,
    app_key,
)


@app.route("/", methods=['POST'])
def bot():
    # Get the event payload
    payload = request.json

    # Check if the event is a GitHub PR creation event
    if not all(k in payload.keys() for k in ['action', 'pull_request']) and \
            payload['action'] == 'opened':
        return "ok"

    owner = payload['repository']['owner']['login']
    repo_name = payload['repository']['name']

    # Get a git connection as our bot
    # Here is where we are getting the permission to talk as our bot and not
    # as a Python webservice
    git_connection = Github(
        login_or_token=git_integration.get_access_token(
            git_integration.get_installation(owner, repo_name).id
        ).token
    )
    repo = git_connection.get_repo(f"{owner}/{repo_name}")

    issue = repo.get_issue(number=payload['pull_request']['number'])

    # Call meme-api to get a random meme
    response = requests.get(url='https://meme-api.herokuapp.com/gimme')
    if response.status_code != 200:
        return 'ok'

    # Get the best resolution meme
    meme_url = response.json()['preview'][-1]
    # Create a comment with the random meme
    issue.create_comment(f"![Alt Text]({meme_url})")
    return "ok"


"""
Create a review for a pull request :
POST Request :

('POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews', {
  owner: 'OWNER',
  repo: 'REPO',
  pull_number: 'PULL_NUMBER',
  commit_id: 'ecdd80bb57125d7ba9641ffaa4d7d2c19d3f3091',
  body: 'This is close to perfect! Please address the suggested inline change.', # main comment of the review! (LGTM for lysandre)
  event: 'REQUEST_CHANGES',
  comments: [ # list of all the comments and suggestions
    {
      path: 'file.md',
      position: 6,
      body: 'Please add more information here, and fix this typo.'
    }
  ]
})

"""


# repo = g.get_repo("PyGithub/PyGithub")
# issue = repo.get_issue(number=874)
# issue.create_comment("Test")
# IssueComment(user=NamedUser(login="user"), id=36763078)

if __name__ == "__main__":
    app.run(debug=True, port=5000)