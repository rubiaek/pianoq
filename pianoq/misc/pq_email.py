import smtplib
import ssl


class Mail(object):
    def __init__(self):
        self.port = 465
        self.smtp_server_domain_name = "smtp.gmail.com"
        self.sender_mail = "ohad.lab1@gmail.com"
        self.password = "ohadlabmail"

    def send(self, content, to='ronen.shekel@mail.huji.ac.il', subject='Automatic mail from lab'):
        ssl_context = ssl.create_default_context()
        service = smtplib.SMTP_SSL(self.smtp_server_domain_name, self.port, context=ssl_context)
        service.login(self.sender_mail, self.password)

        result = service.sendmail(self.sender_mail, to, f"Subject: {subject}\n{content}")

        service.quit()


def send_email(content, to='ronen.shekel@mail.huji.ac.il', subject='Automatic mail from lab'):
    m = Mail()
    m.send(content, to, subject)


