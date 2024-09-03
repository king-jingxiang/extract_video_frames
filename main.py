import os.path

import gradio as gr
import video_extract as ext

VIDEO_FILE_PATH = ""
OUTPUT_EXTRACT_FOLDER = ""


def update_video_path(video_file):
    global VIDEO_FILE_PATH
    VIDEO_FILE_PATH = video_file
    print(f"VIDEO_FILE_PATH: {VIDEO_FILE_PATH}")
    return video_file


def load_video(video_file):
    global VIDEO_FILE_PATH
    if video_file:
        VIDEO_FILE_PATH = video_file
        print(f"VIDEO_FILE_PATH: {VIDEO_FILE_PATH}")
        # return video_file
    # return None


def on_file_select(evt: gr.SelectData, output_path):
    return ext.display_file_content(output_path, os.path.basename(evt.value))


def on_subtitle_select(evt: gr.SelectData, output_path, instruction, prompt, timestamp, join_sep):
    print("on_subtitle_select", os.path.basename(evt.value), output_path, instruction, prompt, timestamp, join_sep)
    return ext.display_prompt_content(output_path, os.path.basename(evt.value), instruction, prompt, timestamp,
                                      join_sep)


def get_iframe_content(url):
    return f'<iframe src="{url}" width="100%" height="600px"></iframe>'


# Gradio界面
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("提取相似帧"):
            with gr.Row():
                gr.Markdown("## 请先上传视频或者指定本地视频文件路径")
            with gr.Row():
                video = gr.Video(label="上传视频")
                with gr.Column():
                    video_path = gr.Textbox(label="指定本地视频文件路径")
                    output_path = gr.Textbox(label="中间结果存储的路径", placeholder="不指定则为当前目录")
            with gr.Row():
                gr.Markdown("## 配置抽帧时间间隔以及相似度计算阈值")
            with gr.Row():
                similarity_algo = gr.Dropdown(choices=["ahash", "phash", "dhash", "whash"], value="ahash",
                                              label="相似度计算算法")
                threshold = gr.Slider(minimum=0, maximum=10, step=1, value=5, label="图片相似度阈值")
                interval_sec = gr.Slider(minimum=0, maximum=10, step=0.5, value=1.5, label="抽帧间隔（秒）")
            with gr.Row():
                extract_frame_button = gr.Button("提取相似帧")
            with gr.Row():
                gr.Markdown("## 输出结果展示")
            with gr.Row():
                frame_images = gr.Gallery(
                    label="提取的相似帧", show_label=False, elem_id="gallery"
                    , columns=[1], rows=[1], object_fit="contain", height="auto")

            # 上传完时候之后，自动更新video_path的值
            video.change(fn=update_video_path, inputs=video, outputs=video_path)
            # 输入框输入完之后，自动加载视频 TODO会和video.change影响，暂时不加了
            # video_path.submit(fn=load_video, inputs=video_path, outputs=video)
            video_path.submit(fn=load_video, inputs=video_path)
            extract_frame_button.click(fn=ext.extract_similar_frames,
                                       inputs=[video_path, output_path, interval_sec, threshold, similarity_algo],
                                       outputs=frame_images)

        with gr.TabItem("提取音频并转换文字"):
            with gr.Row():
                gr.Markdown("## 合并选项")
            with gr.Row():
                merge_small_subtitles = gr.Checkbox(label="是否合并较小的字幕文件")
                file_size_threshold = gr.Slider(minimum=0, maximum=4096, step=1, value=1024, label="文件大小(B)")
            with gr.Row():
                gr.Markdown("## 视频文件和输出信息")
            with gr.Row():
                sub_conv_btn = gr.Button("提取并转换")
            with gr.Row():
                gr.Markdown("## 输出信息")
            with gr.Row():
                sub_file_output = gr.File(label="字幕文件列表", show_label=False, file_types=["txt"])
                sub_frame = gr.Image(label="视频帧")
                sub_file_content = gr.Textbox(label="文件内容")

            sub_conv_btn.click(ext.extract_relative_subtitle_with_merge,
                               inputs=[video_path, output_path, merge_small_subtitles, file_size_threshold],
                               outputs=sub_file_output)
            sub_file_output.select(on_file_select, inputs=[output_path], outputs=[sub_file_content, sub_frame])

        with gr.TabItem("字幕片段总结"):
            with gr.Row():
                gr.Markdown("## 提示词工程")
            with gr.Row():
                max_new_tokens = gr.Slider(minimum=0, maximum=8192, step=1, value=1024, label="max_new_tokens")
                temperature = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.2, label="temperature")
                top_p = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.9, label="top_p")
                top_k = gr.Slider(minimum=0, maximum=100, step=1, value=20, label="top_k")
            with gr.Column():
                instruction = gr.Textbox(label="系统指令", value=ext.DEFAULT_SYSTEM_PROMPT)
                prompt = gr.Textbox(label="提示词", value=ext.DEFAULT_SUBTITLE_SUMMARY_PROMPT)
                with gr.Row():
                    with gr.Column():
                        list_file_btn = gr.Button("列出所有字幕文件")
                        sub_file_lists = gr.File(label="字幕文件列表", show_label=False, file_types=["txt"])
                    with gr.Column():
                        join_sep = gr.Dropdown(choices=["None", "\\n", "\\t", "\" \""], value="None",
                                               label="字幕拼接字符")
                        timestamp = gr.Checkbox(label="添加时间戳")
                        sub_preview_text = gr.Textbox(label="提示词预览")

            with gr.Row():
                # TODO 将prompt和text分开
                sub_summary_btn = gr.Button("生成该片段总结")
            with gr.Row():
                summary_content = gr.Textbox(label="总结输出")
            with gr.Row():
                gen_all = gr.Button("生成所有片段总结")
            with gr.Row():
                gen_file_output = gr.File(label="总结文件列表", show_label=False, file_types=["txt"])
            sub_summary_btn.click(ext.summarize_relative_subtitle,
                                  inputs=[instruction, sub_preview_text, max_new_tokens, temperature, top_p, top_k, ],
                                  outputs=summary_content)
            gen_all.click(ext.summarize_all_relative_subtitle,
                          inputs=[output_path, instruction, prompt, timestamp, join_sep,
                                  max_new_tokens, temperature, top_p, top_k, ],
                          outputs=gen_file_output)
            list_file_btn.click(ext.list_all_subtitles, inputs=[output_path], outputs=sub_file_lists)
            sub_file_lists.select(on_subtitle_select, inputs=[output_path, instruction, prompt, timestamp, join_sep],
                                  outputs=sub_preview_text)

        with gr.TabItem("视频总结"):
            with gr.Row():
                gr.Markdown("## 提示词工程")
            with gr.Row():
                max_new_tokens = gr.Slider(minimum=0, maximum=8192, step=1, value=1024, label="max_new_tokens")
                temperature = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.2, label="temperature")
                top_p = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.9, label="top_p")
                top_k = gr.Slider(minimum=0, maximum=100, step=1, value=20, label="top_k")
            with gr.Column():
                instruction = gr.Textbox(label="系统指令", value=ext.DEFAULT_SYSTEM_PROMPT)
                prompt = gr.Textbox(label="提示词", value=ext.DEFAULT_VIDEO_SUMMARY_PROMPT)
                with gr.Row():
                    summary_preview_btn = gr.Button("预览提示词")
                    video_summary_text = gr.Textbox(label="提示词预览")

            with gr.Row():
                gen_video_summary_btn = gr.Button("生成视频总结")
            with gr.Row():
                summary_content = gr.Textbox(label="总结输出")
            gen_video_summary_btn.click(ext.summarize_video_summary,
                                        inputs=[output_path, instruction, prompt, max_new_tokens, temperature, top_p,
                                                top_k],
                                        outputs=summary_content)
            # TODO 预览
            summary_preview_btn.click(ext.summarize_video_summary_preview, inputs=[output_path, instruction, prompt],
                                      outputs=video_summary_text)
        with gr.TabItem("视频思维导图"):
            with gr.Row():
                gr.Markdown("## 提示词工程")
            with gr.Row():
                max_new_tokens = gr.Slider(minimum=0, maximum=8192, step=1, value=1024, label="max_new_tokens")
                temperature = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.2, label="temperature")
                top_p = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.9, label="top_p")
                top_k = gr.Slider(minimum=0, maximum=100, step=1, value=20, label="top_k")
            with gr.Column():
                instruction = gr.Textbox(label="系统指令", value=ext.DEFAULT_SYSTEM_PROMPT)
                prompt = gr.Textbox(label="提示词", value=ext.DEFAULT_VIDEO_MINDMAP_PROMPT)
                with gr.Row():
                    mind_prompt_preview_btn = gr.Button("预览提示词")
                    mind_prompt_preview_text = gr.Textbox(label="提示词预览")
            with gr.Row():
                gen_mindmap_md_btn = gr.Button("生成思维导图-Markdown")
            with gr.Row():
                mind_summary_content = gr.Textbox(label="总结输出")
            with gr.Row():
                gr.Markdown("### 渲染思维导图")
            with gr.Row():
                gr.HTML(get_iframe_content("https://markmap.js.org/repl"))
            gen_mindmap_md_btn.click(ext.summarize_video_mindmap,
                                     inputs=[output_path, instruction, prompt, max_new_tokens, temperature, top_p,
                                             top_k],
                                     outputs=mind_summary_content)

            mind_prompt_preview_btn.click(ext.summarize_video_summary_preview,
                                          inputs=[output_path, instruction, prompt],
                                          outputs=mind_prompt_preview_text)
        with gr.TabItem("保存输出结果"):
            with gr.Row():
                gr.Markdown("## 最终结果")
            with gr.Row():
                summary_md = gr.Checkbox(label="整体总结",value=True)
            with gr.Row():
                gr.Markdown("## 中间结果")
            with gr.Row():
                subtitle = gr.Checkbox(label="字幕")
                seg_subtitle = gr.Checkbox(label="字幕分片")
                seg_video_summary = gr.Checkbox(label="视频总结")
            with gr.Row():
                save_button = gr.Button("保存输出结果")
            with gr.Row():
                file_output = gr.File(label="输出文件", show_label=False, file_types=["zip"])
            save_button.click(ext.save_output_to_file,
                              inputs=[output_path, subtitle, seg_subtitle, seg_video_summary], outputs=file_output)

demo.launch(server_name="0.0.0.0")
