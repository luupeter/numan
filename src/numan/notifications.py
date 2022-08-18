import urllib3
import json
import traceback
import base64

# read more https://api.slack.com/methods/chat.postMessage
# and here is how to listen to messages: https://slack.dev/bolt-python/concepts !!!

# this is not very smart...I know... please don't post anything to our channel :D
webhook_url = base64.b64decode(
    'aHR0cHM6Ly9ob29rcy5zbGFjay5jb20vc2VydmljZXMvVFBFRkw3QjZDL' \
    '0IwM1UyNllDNDhNL0hhdHAxYUFyVE9JNElsbTA0VWRLVzJIbg=='.encode('ascii')).decode('ascii')

user_id = {"anna": "UPGMQ34BG", "peter": "UPE21JHRA"}


# Send Slack notification based on the given message
def slack_notification(message, tag_users=None):
    if tag_users is not None:
        tags = ""
        for user in tag_users:
            tags = tags + f'<@{user_id[user]}>'
        message = f'{tags}\n {message}'

    try:
        slack_message = {'text': message}

        http = urllib3.PoolManager()
        response = http.request('POST',
                                webhook_url,
                                body=json.dumps(slack_message),
                                headers={'Content-Type': 'application/json'},
                                retries=False)
    except:
        traceback.print_exc()

    return True


if __name__ == "__main__":
    slack_notification(f'go to sleep now ?',
                       tag_users=["anna"])
