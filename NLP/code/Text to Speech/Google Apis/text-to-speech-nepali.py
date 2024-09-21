from gtts import gTTS
import os

# Nepali text you want to convert to speech
# text = """Morning (सुप्रभात):
# सुप्रभात! बिहानको उज्यालो सूर्योदयले दिनको शुभारम्भ गर्दछ। चिडियाहरूको चुँचुं र फूलहरूको सुगन्धले वातावरणलाई ताजगी दिन्छ। बिहानको शान्ति र सजीवता मनलाई आनन्दित बनाउँछ। यस समयमा मानिसहरूले नयाँ ऊर्जा र उत्साहका साथ दिनको योजना बनाउँछन्।

# Evening (साँझ):
# साँझको समयमा सूर्य अस्ताउने क्रममा आकाशमा सुनौलो रंगको छटा फैलिन्छ। यो समय विश्राम र सँगैको क्षणको हुन्छ। मानिसहरू परिवार र साथीहरू सँग भेला भएर दिनका अनुभवहरू साझा गर्छन्। साँझको शान्ति र सौन्दर्यले मनमा खुशीको अनुभूति ल्याउँछ।"""

text = """
डाक्टर: तपाईँको रक्त चिनी स्तरको रिपोर्टमा, उपवासको रक्त चिनी १२० मि.ग्रा./डेलि छ, जुन सामान्यभन्दा थोरै उच्च छ। खाना खाएको पछि स्तर १८० मि.ग्रा./डेलि छ, जुन पनि उच्च छ। यसको अर्थ तपाईँलाई प्रीडायबिटिसको खतरा छ। कृपया ध्यान दिनुहोस् र आहारमा परिवर्तन गर्नुहोस्। नियमित रूपमा रक्त चिनी परीक्षण गरिरहनुहोस्।
"""

# Create a gTTS object, specifying the Nepali language (lang='ne')
tts = gTTS(text=text, lang='ne')

# Save the converted audio to a file
tts.save("nepali_voice.mp3")

# Play the saved audio file using system's default player
os.system("start nepali_voice.mp3")  # For Windows
# os.system("afplay nepali_voice.mp3")  # For macOS
# os.system("mpg321 nepali_voice.mp3")  # For Linux


from gtts import gTTS
import os
from playsound import playsound
import tempfile

# Nepali text you want to convert to speech
text = """
डाक्टर: तपाईँको रक्त चिनी स्तरको रिपोर्टमा, उपवासको रक्त चिनी १२० मि.ग्रा./डेलि छ, जुन सामान्यभन्दा थोरै उच्च छ। खाना खाएको पछि स्तर १८० मि.ग्रा./डेलि छ, 
जुन पनि उच्च छ। यसको अर्थ तपाईँलाई प्रीडायबिटिसको खतरा छ। कृपया ध्यान दिनुहोस् र आहारमा परिवर्तन गर्नुहोस्। नियमित रूपमा रक्त चिनी परीक्षण गरिरहनुहोस्।
"""

# Create a gTTS object, specifying the Nepali language (lang='ne')
tts = gTTS(text=text, lang='ne')

# Create a temporary file
with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
    # Save the converted audio to a temporary file
    tts.save(tmpfile.name)
    
    # Play the saved audio file
    playsound(tmpfile.name)


