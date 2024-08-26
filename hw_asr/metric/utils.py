import editdistance

# Don't forget to support cases when target_text == ''

def calc_cer(target_text, predicted_text) -> float:
    if target_text == '':
        return 0.0 if predicted_text == '' else 1.0
    return editdistance.eval(target_text, predicted_text) 


def calc_wer(target_text, predicted_text) -> float:
    if target_text == '':
        return 0.0 if predicted_text == '' else 1.0
    target_words = target_text.split()
    predicted_words = predicted_text.split()
    return editdistance.eval(target_words, predicted_words) 