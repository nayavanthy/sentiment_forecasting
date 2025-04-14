from google.auth import default

creds, project = default()
print(f"Authenticated as project: {project}")
