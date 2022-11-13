import random
import requests


def get_kateglo(phrase):
    kateglo_url ='http://kateglo.lostfocus.org/api.php?format=json&phrase='
    query = kateglo_url + phrase
    
    req_response = requests.get(query)
    try:
        data = req_response.json()
    except Exception as err:
        return None

    return data

def random_pick_phrase(t_ori, n):
    list_phrase = t_ori.split()
    list_len = len(list_phrase)
    
    phrases = []
    if list_len >= n:
        lp = set(list_phrase) #to enable unique set list
        phrases = random.sample(lp, n)
        return phrases

    return None

#Mendefinisikan Fungsi Generated Text
def generate_text(text_ori, n = 1):
    t_ori = text_ori
    generated_text = []
    while n > 0:
        status_not_found = True
        while status_not_found:
            word_be_changed = random_pick_phrase(text_ori, 1)
            if word_be_changed is None:
                return(generated_text)

            for i in word_be_changed:
                json_dict = get_kateglo(i)

                if(json_dict is not None):
                    try:
                        for rel in json_dict['kateglo']['all_relation'] :
                            if rel.get('rel_type') == 's':       
                                gT=t_ori.replace(i, rel.get('related_phrase'))            
                                generated_text.append(gT)
                                text_ori = text_ori.replace(i, "")
                                status_not_found = False
                                break
                        if(status_not_found):
                            text_ori = text_ori.replace(i, "")
                    except:
                        text_ori = text_ori.replace(i, "")
                        status_not_found=True

                else:
                    text_ori = text_ori.replace(i, "")
                    status_not_found=True
        n = n-1
      
    return(generated_text)