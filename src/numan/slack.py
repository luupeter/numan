import urllib3
import json
import traceback

# read more https://api.slack.com/methods/chat.postMessage
# and here is how to listen to messages: https://slack.dev/bolt-python/concepts !!!

# this is not very smart...I know... please don't post anything to our channel :D
webhook_url = '^%^h^#%t%#%^t%#%p^^%#%s%#:#/#/#%^h%#%^o%#%^o%#%^k%#%^s%#.#%^s%#%%' \
              '^l%#%^a%#%^c%#%^k%#.#%^c%#%^o%#%^m%#/#%^s%#%^e%#%^r%#%^v%#%^i%#%^c%#' \
              '%^e%#%^s%#/#%^T%#%^P%#%^E%#%^F%#%^L%#%^7%#%^B%#%^6%#%^C%#/#%^B%#%^0%#%^' \
              '3%#%^U%#%^2%#%^6%#%^Y%#%^C%#%^4%#%^8%#%^M%#/#%^H%#%^a%#%^t%#%^p%#%^1%#%a%' \
              '#%A%#%r%#%T%#%O%#%I%#%4%#%I%#%l%#%m%#%0%#%4%#%U%#%d%#%K%#%W%#%2%#%H%#%%n%'
webhook_url = webhook_url.replace('^', '')
webhook_url = webhook_url.replace('%', '')
webhook_url = webhook_url.replace('#', '')

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
    slack_notification(f'Will go to sleep now ......',
                       tag_users=["peter"])
