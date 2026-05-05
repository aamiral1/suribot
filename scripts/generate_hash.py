import bcrypt
import getpass

pw = getpass.getpass("Password: ")
print(bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode())
