import os
import smtplib
import markdown
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.header import Header
from datetime import datetime
from pathlib import Path
import pytz  # 添加时区支持

class Mailer:
    def __init__(self):
        # 优先从环境变量读取，如果没有则从配置文件读取
        try:
            from config.settings import MAIL_CONFIG
            self.smtp_server = os.getenv("SMTP_SERVER") or MAIL_CONFIG.get('smtp_server', "smtp.feishu.cn")
            self.smtp_port = int(os.getenv("SMTP_PORT") or str(MAIL_CONFIG.get('smtp_port', 465)))
            self.sender_email = os.getenv("SENDER_EMAIL") or MAIL_CONFIG.get('sender_email')
            self.sender_password = os.getenv("SENDER_PASSWORD") or MAIL_CONFIG.get('sender_password')
            self.receiver_email = os.getenv("RECEIVER_EMAIL") or MAIL_CONFIG.get('receiver_email', "xiaojingze@comein.cn")
        except ImportError:
            # 如果导入失败，使用环境变量或默认值
            self.smtp_server = os.getenv("SMTP_SERVER", "smtp.feishu.cn")
            self.smtp_port = int(os.getenv("SMTP_PORT", "465"))
            self.sender_email = os.getenv("SENDER_EMAIL", "xiaojingze@comein.cn")
            self.sender_password = os.getenv("SENDER_PASSWORD")
            self.receiver_email = os.getenv("RECEIVER_EMAIL", "xiaojingze@comein.cn")
        
        # 使用北京时区
        self.beijing_tz = pytz.timezone('Asia/Shanghai')

    def _get_beijing_date(self):
        """获取北京时区的当前日期"""
        beijing_time = datetime.now(self.beijing_tz)
        return beijing_time.strftime('%Y-%m-%d')

    def send_daily_summary(self, file_path):
        if not all([self.sender_email, self.sender_password, self.receiver_email]):
            print("⚠️ 邮件配置不完整，跳过发送步骤。")
            print(f"   发件人: {self.sender_email or '未设置'}")
            print(f"   收件人: {self.receiver_email or '未设置'}")
            print(f"   密码: {'已设置' if self.sender_password else '未设置（需要设置 SENDER_PASSWORD 环境变量）'}")
            print("   提示: 请设置环境变量 SENDER_PASSWORD（飞书邮箱的应用专用密码）")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_text = f.read()

            import re
            file_dir = Path(file_path).parent
            
            # 预处理：确保列表项在markdown中有正确的换行
            # markdown文件中的列表项应该已经是分开的，这里主要是确保格式正确
            
            # 先转换为HTML，然后处理图片
            html_body = markdown.markdown(md_text, extensions=['extra'])
            
            # 处理图片：查找HTML中的所有图片标签（包括markdown转换后的和HTML div中的）
            img_pattern = r'<img\s+[^>]*src=[\'"]([^\'"]+)[\'"][^>]*>'
            img_matches = list(re.finditer(img_pattern, html_body))
            
            # 将图片路径替换为cid引用，并准备附件
            img_cids = {}
            for match in img_matches:
                img_path = match.group(1)
                # 跳过已经是cid的图片
                if img_path.startswith('cid:'):
                    continue
                    
                # 如果是相对路径，转换为绝对路径
                if not os.path.isabs(img_path):
                    full_img_path = file_dir / img_path
                else:
                    full_img_path = Path(img_path)
                
                if full_img_path.exists():
                    # 生成唯一的cid
                    cid = f"img_{len(img_cids)}"
                    img_cids[cid] = str(full_img_path)
                    # 替换img标签中的src为cid（使用更精确的替换）
                    original_img_tag = match.group(0)
                    new_img_tag = re.sub(r'src=[\'"]([^\'"]+)[\'"]', f'src="cid:{cid}"', original_img_tag)
                    html_body = html_body.replace(original_img_tag, new_img_tag, 1)
                else:
                    print(f"⚠️ 图片文件不存在: {full_img_path}")
            
            # 后处理：修复格式问题
            # 0. 先处理"赛道观察"部分：分离引用块内的列表项
            # 如果引用块内包含列表项模式（文本后跟着 " - " 或 "\n- "），需要分离
            def separate_blockquote_list(match):
                blockquote_full = match.group(0)
                blockquote_content = match.group(1)
                
                # 检查引用块内容中是否包含列表项模式
                # 查找 " - " 或 "\n- " 后面跟着大写字母开头的文本（论文标题）
                # 这种情况说明列表项被错误地包含在引用块内
                if re.search(r'[^\n]\s+-\s+[A-Z]', blockquote_content) or re.search(r'\n\s*-\s+[A-Z]', blockquote_content):
                    # 分离：找到第一个 " - " 或 "\n- " 作为分隔点
                    # 将引用块内容分为文本部分和列表项部分
                    # 使用更精确的匹配：查找 " - " 后面跟着大写字母（论文标题）
                    match_obj = re.search(r'([^\n]+?)\s+(-\s+[A-Z][^\n]*(?:\n\s*-\s+[A-Z][^\n]*)*)', blockquote_content, re.DOTALL)
                    if match_obj:
                        text_part = match_obj.group(1).strip()  # 文本部分
                        list_items_text = match_obj.group(2)  # 列表项文本（包含 "- " 前缀）
                        
                        # 将列表项文本转换为多个 <li> 标签
                        list_items = re.findall(r'-\s+([^\n]+)', list_items_text)
                        # 确保每个列表项之间有换行，但不要有额外的空白
                        list_items_html = '<br>'.join([f'<li>{item.strip()}</li>' for item in list_items])
                        
                        return f'<blockquote><p>{text_part}</p></blockquote><ul class="blockquote-list">{list_items_html}</ul>'
                
                return blockquote_full
            
            # 匹配引用块，检查是否需要分离列表项（在步骤2之前处理）
            html_body = re.sub(r'<blockquote>(.*?)</blockquote>', separate_blockquote_list, html_body, flags=re.DOTALL)
            
            # 1. 修复论文信息字段之间的多余空行（让它们像自然换行）
            # Markdown转换后，每个字段（- **中文标题**: 等）可能是独立的 <p> 标签
            # 需要将这些字段的间距减小到最小，让它们看起来像连续的行
            
            # 将所有连续的 <p> 标签之间的内容替换为 <br>，确保字段之间没有空行
            # 使用更精确的匹配，包括可能存在的空白字符和换行符
            html_body = re.sub(r'</p>\s*\n?\s*<p>', r'<br>', html_body)
            
            # 清理多余的 <br>（连续多个只保留一个）
            html_body = re.sub(r'<br>\s*<br>+', r'<br>', html_body)
            html_body = re.sub(r'<br>\s*\n\s*<br>', r'<br>', html_body)
            
            # 2. 处理列表项：markdown库可能将列表项转换为 <p> 标签而不是 <ul><li>
            # 查找所有以 "- " 开头的 <p> 标签，将它们转换为列表项
            
            # 使用更精确的正则匹配，处理可能包含HTML标签的情况
            # 匹配 <p>- 开头，</p> 结尾的段落（包括可能包含HTML标签的内容）
            def convert_p_to_li(match):
                content = match.group(1)
                # 移除开头的 "- " 和可能的空格
                content = re.sub(r'^-\s*', '', content.strip())
                return f'<li>{content}</li>'
            
            # 处理包含HTML标签的列表项（如包含 <strong> 等）
            # 使用非贪婪匹配和多行模式
            html_body = re.sub(r'<p>(-[^<]*(?:<[^>]+>[^<]*</[^>]+>[^<]*)*?)</p>', convert_p_to_li, html_body, flags=re.DOTALL)
            
            # 将连续的 <li> 包裹在 <ul> 中
            # 查找连续的列表项（可能被空白或 <br> 分隔），将它们包裹在 <ul> 中
            def wrap_li_sequence(match):
                content = match.group(0)
                # 确保每个 <li> 之间有换行
                content = re.sub(r'</li>\s*<li>', r'</li><br><li>', content)
                return f'<ul>{content}</ul>'
            
            # 匹配连续的列表项（至少2个），它们之间可能有空白或 <br>
            # 更稳健：基于标签而不是内容
            html_body = re.sub(
                r'((?:<li>.*?</li>\s*){2,})',
                lambda m: f"<ul>{m.group(1)}</ul>",
                html_body,
                flags=re.DOTALL
            )

            # 3. 确保列表项正确换行
            # 先移除列表项内容中的 <p> 标签（如果存在）
            html_body = re.sub(r'<li>\s*<p>', r'<li>', html_body)
            html_body = re.sub(r'</p>\s*</li>', r'</li>', html_body)
            
            # 4. 先处理"赛道观察"部分的列表项（在blockquote后面），确保它们之间有换行但不要空行
            # 对于blockquote后面的ul列表，确保每个列表项之间有<br>换行
            def process_blockquote_list(match):
                ul_content = match.group(1)
                # 清理列表项内容中的多余空白
                ul_content = re.sub(r'<li>\s+', r'<li>', ul_content)
                ul_content = re.sub(r'\s+</li>', r'</li>', ul_content)
                # 如果列表项之间没有<br>，添加一个（确保每个列表项后都有换行）
                ul_content = re.sub(r'</li>(?!\s*<br>)\s*<li>', r'</li><br><li>', ul_content)
                # 确保列表项之间只有一个<br>，移除多余的<br>和空白
                ul_content = re.sub(r'</li>\s*<br>\s*<br>\s*<li>', r'</li><br><li>', ul_content)
                ul_content = re.sub(r'</li>\s+<br>\s+<li>', r'</li><br><li>', ul_content)
                
                # 添加特殊标记，表示这个ul已经被处理过了
                return f'</blockquote><ul data-processed="blockquote" class="blockquote-list">{ul_content}</ul>'
            
            # 匹配blockquote后面的ul列表（"赛道观察"部分），添加特殊标记以便后续识别
            html_body = re.sub(r'</blockquote>\s*<ul>(.*?)</ul>', process_blockquote_list, html_body, flags=re.DOTALL)
            
            # 5. 对于论文信息字段的列表项（包含 "**" 的，如 **中文标题**:），移除它们之间的<br>
            # 让它们像自然换行一样显示，没有空行
            def process_info_list(match):
                ul_full = match.group(0)
                # 如果这个ul已经被处理过了（blockquote后面的），跳过
                if 'data-processed="blockquote"' in ul_full:
                    return ul_full
                
                ul_content = match.group(1)
                # 如果列表项内容包含 "**"（论文信息字段），移除它们之间的<br>
                if '**' in ul_content:
                    ul_content = re.sub(r'</li><br><li>', r'</li><li>', ul_content)
                    ul_content = re.sub(r'</li>\s*<br>\s*<li>', r'</li><li>', ul_content)
                return f'<ul>{ul_content}</ul>'
            
            # 匹配所有ul列表，处理论文信息字段
            # 排除已经被处理过的blockquote后面的ul
            html_body = re.sub(
                r'<ul(?![^>]*class="blockquote-list")(.*?)</ul>',
                process_info_list,
                html_body,
                flags=re.DOTALL
            )
            
            # 移除列表结束标签前的多余 <br>（但保留列表项之间的<br>）
            html_body = re.sub(r'<br>\s*</ul>', r'</ul>', html_body)
            html_body = re.sub(r'<br>\s*</ol>', r'</ol>', html_body)
            
            # 清理临时标记（但保留class属性，用于CSS样式）
            html_body = re.sub(r' data-processed="blockquote"', '', html_body)
                        # 移除孤立的 blockquote 遗留符号
            html_body = re.sub(r'(?m)^\s*&gt;\s*$', '', html_body)
            html_body = re.sub(r'(?m)^\s*>\s*$', '', html_body)
            
            
            # 添加CSS样式确保列表项正确换行显示
            styled_html = f"""<html>
<head>
<style>
    body {{
        font-family: Arial, sans-serif;
        line-height: 1.6;
    }}
    ul, ol {{
        margin: 10px 0;
        padding-left: 30px;
        list-style: none;
    }}
    li {{
        margin: 0;
        display: block !important;
        line-height: 1.5;
        padding-left: 20px;
        position: relative;
        margin-bottom: 0;
    }}
    li::before {{
        content: "•";
        position: absolute;
        left: 0;
        color: #333;
    }}
    /* 论文信息字段的列表项之间不应该有间距 */
    ul li, ol li {{
        margin-bottom: 0;
        margin-top: 0;
    }}
    /* "赛道观察"部分的列表项（论文标题）之间需要换行，但不要空行 */
    blockquote {{
        margin: 0 !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
    }}
    blockquote p {{
        margin: 0 !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
    }}
    blockquote + ul.blockquote-list {{
        margin: 0 !important;
        padding: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }}
    ul.blockquote-list {{
        margin: 0 !important;
        padding: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }}
    ul.blockquote-list li {{
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        line-height: 1.4 !important;
        padding-left: 20px !important;
    }}
    ul.blockquote-list br {{
        line-height: 1.4 !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 0 !important;
        font-size: 0 !important;
        display: block !important;
        content: "" !important;
    }}
    blockquote + ul.blockquote-list li, blockquote + ul.blockquote-list li + li {{
        display: block !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }}
    p {{
        margin: 0 !important;
        line-height: 1.4;
        padding: 0;
    }}
    /* 确保段落之间没有间距 */
    p + p {{
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }}
    /* 确保 <br> 标签不会产生额外间距 */
    br {{
        line-height: 1.4;
        margin: 0;
        padding: 0;
    }}
    h2, h3, h4 {{
        margin: 15px 0 10px 0;
    }}
    /* 减少列表项之间的间距 */
    ul li + li, ol li + li {{
        margin-top: 0;
    }}
</style>
</head>
<body>{html_body}</body>
</html>"""
            
            # 解析收件人列表
            receivers = [r.strip() for r in self.receiver_email.split(',')]
            
            # 建立一次连接，循环发送给多个人
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            server.login(self.sender_email, self.sender_password)
            
            # 使用北京时区的日期
            beijing_date = self._get_beijing_date()
            
            for recipient in receivers:
                try:
                    msg = MIMEMultipart()
                    msg['From'] = self.sender_email
                    msg['To'] = recipient  # 关键：这里只写当前这一个人的地址
                    msg['Subject'] = Header(f"Arxiv LLM Daily 研报 - {beijing_date}", 'utf-8')
                    
                    # 添加HTML正文
                    msg.attach(MIMEText(styled_html, 'html', 'utf-8'))
                    
                    # 添加图片附件
                    for cid, img_path in img_cids.items():
                        try:
                            with open(img_path, 'rb') as img_file:
                                img_data = img_file.read()
                            img = MIMEImage(img_data)
                            img.add_header('Content-ID', f'<{cid}>')
                            img.add_header('Content-Disposition', 'inline', filename=Path(img_path).name)
                            msg.attach(img)
                        except Exception as img_e:
                            print(f"⚠️ 添加图片 {img_path} 失败: {img_e}")
                    
                    server.sendmail(self.sender_email, recipient, msg.as_string())
                    print(f"✅ 邮件已成功单发至: {recipient}")
                except Exception as inner_e:
                    print(f"❌ 向 {recipient} 发送失败: {inner_e}")
            
            server.quit()
        except Exception as e:
            print(f"❌ 邮件发送流程出错: {e}")
    
    def send_no_papers_message(self):
        """发送没有新论文的消息"""
        if not all([self.sender_email, self.sender_password, self.receiver_email]):
            print("⚠️ 邮件配置不完整，跳过发送步骤。")
            print(f"   发件人: {self.sender_email or '未设置'}")
            print(f"   收件人: {self.receiver_email or '未设置'}")
            print(f"   密码: {'已设置' if self.sender_password else '未设置（需要设置 SENDER_PASSWORD 环境变量）'}")
            print("   提示: 请设置环境变量 SENDER_PASSWORD（飞书邮箱的应用专用密码）")
            return
        
        try:
            message = "今天没有新的论文，休息一下吧 😊"
            # 使用北京时区的日期
            beijing_date = self._get_beijing_date()
            html_body = f"""
            <html>
            <body style='font-family: Arial, sans-serif; padding: 20px; text-align: center;'>
                <h2 style='color: #666;'>{message}</h2>
                <p style='color: #999; font-size: 14px;'>Arxiv LLM Daily - {beijing_date}</p>
            </body>
            </html>
            """
            
            # 解析收件人列表
            receivers = [r.strip() for r in self.receiver_email.split(',')]
            
            # 建立一次连接，循环发送给多个人
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            server.login(self.sender_email, self.sender_password)
            
            for recipient in receivers:
                try:
                    msg = MIMEMultipart()
                    msg['From'] = self.sender_email
                    msg['To'] = recipient
                    msg['Subject'] = Header(f"Arxiv LLM Daily - {beijing_date} (无新论文)", 'utf-8')
                    msg.attach(MIMEText(html_body, 'html', 'utf-8'))
                    
                    server.sendmail(self.sender_email, recipient, msg.as_string())
                    print(f"✅ 无新论文通知已发送至: {recipient}")
                except Exception as inner_e:
                    print(f"❌ 向 {recipient} 发送失败: {inner_e}")
            
            server.quit()
        except Exception as e:
            print(f"❌ 邮件发送流程出错: {e}")