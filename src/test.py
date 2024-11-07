from emo_rec import EmoRec

def test():

    try:
        
        import pickle
        # file_content_small file_content_all_light
        with open('./history/pkls/file_content_small.pkl', 'rb') as file: 
            pkl_file = pickle.load(file)
        
        o = EmoRec()
        
        video_res, path_to_res_video, area_plot_res_html, area_plot_png  = o.make_video(pkl_file, 12, 120, 0.5, True, 20, 0.05)
        
        # BytesIO save to file
        with open('area_plot_res_html.html', 'wb') as f:
            f.write(area_plot_res_html)
        # with open('area_plot_png.png', 'wb') as f:
        #     f.write(area_plot_png.getvalue())
        
        t = input()
        
    except Exception as e:
        print(e)



test()
