import sys
from openai import OpenAI
import pickle
from tqdm import tqdm

try:
    sys.path.insert(0,'../../../../Documents/0_perm/keys')
    from key import key
    import os
    os.environ['OPENAI_API_KEY'] = key
except:
    print('Warning when loading openai_utils: Could not find the OpenAI key in the usual location. Please make sure to set manually the OPENAI_API_KEY environment variable!')


GPTClient_DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant.'
class GPTClient:
    def __init__(self):
        self.client = OpenAI()

    def _prompt_is_appropriate(self, input):
        response = self.client.moderations.create(input=input)
        output = response.results[0]
        return output

    def get_response(self, prompt, system_prompt=GPTClient_DEFAULT_SYSTEM_PROMPT, temperature=0, max_tokens=256, model="gpt-4o-mini", max_prompt=4096):

        if not self._prompt_is_appropriate(prompt):
            return 'Inappropriate prompt!'

        response = self.client.chat.completions.create(
                    model = model,
                    temperature = temperature,
                    max_tokens = max_tokens,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "system", "content": system_prompt} #https://community.openai.com/t/the-system-role-how-it-influences-the-chat-behavior/87353
                    ]
                )

        return response.choices[0].message.content


GPTClientForTranslation_SYSTEM_PROMPT = 'You are a translation bot.'
GPTClientForTranslation_PROMPT_TEMPLATE = ('Translate the following text into English. Answer only with the translation. If the text is already in English, answer with the same text.\n'
                '###\n'
                'Text: "{}"\n'
                'Translation: ')

class GPTClientForTranslation(GPTClient):
    def translate(self, text):
        text = text.replace('"', '')
        prompt = GPTClientForTranslation_PROMPT_TEMPLATE.format(text)
        response = self.get_response(
            prompt, system_prompt=GPTClientForTranslation_SYSTEM_PROMPT,
            temperature=0.5, max_tokens=256, model="gpt-4o-mini"
        )
        response = response.replace('"', '')

        return response

    def batch_translate(self, text_batch, tmp_file = 'translations_tmp.pkl'):
        translations = {}
        for i, text in tqdm(enumerate(text_batch), total=len(text_batch)):
            translations[text] = self.translate(text)
            if i % 500 == 0:
                with open(tmp_file, 'wb') as fout:
                    pickle.dump(translations, fout)

        return translations
