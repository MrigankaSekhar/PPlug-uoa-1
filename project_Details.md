Detail of the repository and project 


## Base model key files under the code/ directory
1. aggr_id.py is used in the old model to create files are are used during training
{
  "input": ...,    # original input question
  "his_id": [...], # IDs of historical profile items
  "id": ...,       # question ID
  "output": ...    # the expected answer
}
2. run_all_t5-slim-GNN-mac.sh is used for the model training 
3. here is how the traing file look like 
dev_profile.json:
{"input": "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: Elephant Moon is set in Rangoon, Burma (Myanmar) from December 1941 (after Japanese forces bombed the American naval base at Pearl Harbour near Honolulu in Hawaii) to early 1942. English school teacher Grace Collins has been working in Burma for 20 years. Four thousand people are killed during the Japanese Christmas Day air raid on Rangoon, but the school and everyone in it were safe. As the Japanese Imperial Army prepare to enter Burma, the British colonial rulers and the Americans prepare to evacuate.\n\nHowever, 62 school children  half-castes of Burmese women and foreign men  are not on any countrys evacuation list, nor are they acknowledged by Burma. They are to be abandonned, with no-one to look after them. Grace decides to lead the 62 children to the safety of British-ruled India, 1,000 miles (1,600 kilometres) away  through jungles, mountains, and rivers in the tropical rain  with malaria, illnesses, and the threat of Japanese soldiers.\n\nAt first, their journey was in an old bus, until they see a herd of 53 elephants going to India  45 adults and eight calves. Grace and the children travel with Sam Metcalf, formerly of the Burma Teak Corporation, and a handful of elephant men transporting the elephants across the border to India on behalf of the Burmese Ministry of Agriculture. The children make a game of it by naming the elephants, and these activities juxtapose the brutality of death and murders on route.\n\nThis is fiction, but it is based upon truth  elephant men rescuing refugees from Burma in 1942 after the Fall of Rangoon. Not well-written (for a former BBC reporter), but a nice story about the dignity of elephants and the brutality of humans, as well as the sheer perseverence of the small band of evacuating travellers through extreme challenges.", "his_id": [1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863], "id": "91957", "output": "3"}
{"input": "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: When my go to light went missing in action, I decided to try the Streamlight ProTac HL 750 as a replacement, based on that company's good reputation, but I was so disappointed with what I received that I packaged it up and sent it right back.\n\nThe first thing that I noticed was the lightweight feel of the flashlight. The chassis walls are thin and do not inspire confidence. The threads are coarse and gritty, even after careful cleaning.\n\nAs one other reviewer noted, CR123 batteries rattle noisily within the flashlight body. This annoyance is not a result of any defect or bad design; it is the result of CR123s being slightly narrower than the 18650 rechargeable batteries this flashlight can also use. Flashlight manufacturers who make their lights wide enough for 18650s do so knowing that the CR123s will be smaller. However, many manufacturers deal with this by either including a plastic sheath to use with CR123s to prevent rattling, or engineer the spring tension within their flashlights to do so. Streamlight chose to do neither, thus the annoying rattle. Is this a big deal? Of course not, and if it had been the only thing that I did not like about this light I would have fashioned a sheath of my own, or wrapped a little tape around CR123s to silence them, and carried on.\n\nBut this was not the only thing that I did not like. Which brings me to:\n\nBeam quality. The Streamlight 88040 ProTac HL 750 beam consists of a tight hot spot with a ringy, inconsistent spill. The tight hot spot is neither good nor bad; that is just a matter of taste and preference. However, the ringy corona is a different story. It is not terrible, nor is it bad enough to have make my decision to return the flashlight. It was just the last straw. Back the light went.", "his_id": [1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872], "id": "91556", "output": "3"}

train_aug_input.json:
{"input": "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: I have been using this product for about a month now and I am very satisfied with its performance. The sound quality is excellent and it is very comfortable to wear for long periods of time. The battery life is also impressive, lasting up to 8 hours on a single charge. Overall, I would highly recommend this product to anyone in need of a reliable pair of headphones.", "his_id": [1001, 1002, 1003], "id": "90001", "output": "5"}
{"input": "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: I recently purchased this blender and I am extremely disappointed with its performance. It struggles to blend even soft fruits and vegetables, leaving chunks behind. The motor also seems weak and overheats quickly. Additionally, the design is bulky and takes up too much counter space. I would not recommend this blender to anyone looking for a reliable kitchen appliance.", "his_id": [1004, 1005, 1006], "id": "90002", "output": "2"}

## Training the model
To train the model, you can use the provided shell script `run_all_t5-slim-GNN-mac.sh`. This script is designed to automate the training process using the specified configurations and datasets.
1. the dataset is created using code/PersonalDataset_profile.py
2. the model is defined in code/ModelForPer_slim.py
3. code/main_profile.py is used ad the entry point for training 

## embeddings for the base model 
1. embedding.py which is in the project root is used for creation of the embeddings . the files tha are used for thise are 

dev_questions.json and train_questions.json the structure of these files are as below
[
  {
    "id": "91261",
    "input": "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: My husband is vegan gluten free.  This combination makes it difficult to purchase food that tastes good.  When I found this on this site, I decided to order it for him to try.  He does like \"go veggie\" cheese products that we've found at the local store.  When it arrived, I decided to make something so he could try it.  When we opened the package, the smell of \"parmesan\" filled the air!  It smelled like the real thing!  He tasted it and was very pleased!  He said it tasted like the real thing!  He's happy, which makes me happy!  Thank you \"Go Veggie\" for all your products for Vegan/GF eaters!",
    "profile": [
      {
        "id": "976400",
        "text": "I searched and searched for this product in the stores. I could not find any, but was very excited when I found them here! I like these better than the new alternatives. These are great!",
        "score": "5",
        "date": "2012-03-06"
      },
      {
        "id": "976401",
        "text": "We bought this for a gift for our nephew.  He said he loved it and it is a great bag!",
        "score": "5",
        "date": "2013-01-23"
      }, .... ] , "user_id": 9013773}

## extension model key files under the extension/ directory

1. aggr_id-slim.py is used in the old model to create files are are used during training
{
  "input": ...,    # original input question
  "his_id": [...], # IDs of historical profile items
  "id": ...,       # question ID
  "output": ...    # the expected answer
}
2. run_all_t5-slim-GNN-mac-ext.sh is used for the model training
3. here is how the training files look like 
## training files
dev_profile.json:
{"input": "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: Elephant Moon is set in Rangoon, Burma (Myanmar) from December 1941 (after Japanese forces bombed the American naval base at Pearl Harbour near Honolulu in Hawaii) to early 1942. English school teacher Grace Collins has been working in Burma for 20 years. Four thousand people are killed during the Japanese Christmas Day air raid on Rangoon, but the school and everyone in it were safe. As the Japanese Imperial Army prepare to enter Burma, the British colonial rulers and the Americans prepare to evacuate.\n\nHowever, 62 school children  half-castes of Burmese women and foreign men  are not on any countrys evacuation list, nor are they acknowledged by Burma. They are to be abandonned, with no-one to look after them. Grace decides to lead the 62 children to the safety of British-ruled India, 1,000 miles (1,600 kilometres) away  through jungles, mountains, and rivers in the tropical rain  with malaria, illnesses, and the threat of Japanese soldiers.\n\nAt first, their journey was in an old bus, until they see a herd of 53 elephants going to India  45 adults and eight calves. Grace and the children travel with Sam Metcalf, formerly of the Burma Teak Corporation, and a handful of elephant men transporting the elephants across the border to India on behalf of the Burmese Ministry of Agriculture. The children make a game of it by naming the elephants, and these activities juxtapose the brutality of death and murders on route.\n\nThis is fiction, but it is based upon truth  elephant men rescuing refugees from Burma in 1942 after the Fall of Rangoon. Not well-written (for a former BBC reporter), but a nice story about the dignity of elephants and the brutality of humans, as well as the sheer perseverence of the small band of evacuating travellers through extreme challenges.", "his_id": [1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863], "id": "91957", "output": "3"}
{"input": "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: When my go to light went missing in action, I decided to try the Streamlight ProTac HL 750 as a replacement, based on that company's good reputation, but I was so disappointed with what I received that I packaged it up and sent it right back.\n\nThe first thing that I noticed was the lightweight feel of the flashlight. The chassis walls are thin and do not inspire confidence. The threads are coarse and gritty, even after careful cleaning.\n\nAs one other reviewer noted, CR123 batteries rattle noisily within the flashlight body. This annoyance is not a result of any defect or bad design; it is the result of CR123s being slightly narrower than the 18650 rechargeable batteries this flashlight can also use. Flashlight manufacturers who make their lights wide enough for 18650s do so knowing that the CR123s will be smaller. However, many manufacturers deal with this by either including a plastic sheath to use with CR123s to prevent rattling, or engineer the spring tension within their flashlights to do so. Streamlight chose to do neither, thus the annoying rattle. Is this a big deal? Of course not, and if it had been the only thing that I did not like about this light I would have fashioned a sheath of my own, or wrapped a little tape around CR123s to silence them, and carried on.\n\nBut this was not the only thing that I did not like. Which brings me to:\n\nBeam quality. The Streamlight 88040 ProTac HL 750 beam consists of a tight hot spot with a ringy, inconsistent spill. The tight hot spot is neither good nor bad; that is just a matter of taste and preference. However, the ringy corona is a different story. It is not terrible, nor is it bad enough to have make my decision to return the flashlight. It was just the last straw. Back the light went.", "his_id": [1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872], "id": "91556", "output": "3"}

train_aug_input.json:
{"input": "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: I have been using this product for about a month now and I am very satisfied with its performance. The sound quality is excellent and it is very comfortable to wear for long periods of time. The battery life is also impressive, lasting up to 8 hours on a single charge. Overall, I would highly recommend this product to anyone in need of a reliable pair of headphones.", "his_id": [1001, 1002, 1003], "id": "90001", "output": "5"}
{"input": "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: I recently purchased this blender and I am extremely disappointed with its performance. It struggles to blend even soft fruits and vegetables, leaving chunks behind. The motor also seems weak and overheats quickly. Additionally, the design is bulky and takes up too much counter space. I would not recommend this blender to anyone looking for a reliable kitchen appliance.", "his_id": [1004, 1005, 1006], "id": "90002", "output": "2"}

## Training the model
To train the model, you can use the provided shell script `run_all_t5-slim-GNN-mac.sh`. This script is designed to automate the training process using the specified configurations and datasets.
1. the dataset is created using extention/PersonalDataset_profile_GNN.py
2. the model is defined in extention/ModelForPer_slim_GNN.py
3. extention/main_profile-slim-GNN.py is used ad the entry point for training 
4. this also used a graph model 
5. graph is created and trained using compute_graph_emb_generic_npy.py , this uses dev and train _questions data set
