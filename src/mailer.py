import os
import smtplib
import markdown
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from datetime import datetime
import pytz  # 添加时区支持

class Mailer:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.feishu.cn")
        self.smtp_port = int(os.getenv("SMTP_PORT", "465"))
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.sender_password = os.getenv("SENDER_PASSWORD")
        self.receiver_email = os.getenv("RECEIVER_EMAIL")
        # 使用北京时区
        self.beijing_tz = pytz.timezone('Asia/Shanghai')

    def _get_beijing_date(self):
        """获取北京时区的当前日期"""
        beijing_time = datetime.now(self.beijing_tz)
        return beijing_time.strftime('%Y-%m-%d')

    def send_daily_summary(self, file_path):
        if not all([self.sender_email, self.sender_password, self.receiver_email]):
            print("⚠️ 邮件配置不完整，跳过发送步骤。")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_text = f.read()

            html_body = markdown.markdown(md_text, extensions=['extra'])
            styled_html = f"<html><body style='font-family: Arial, sans-serif;'>{html_body}</body></html>"
            
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
                    msg.attach(MIMEText(styled_html, 'html', 'utf-8'))
                    
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