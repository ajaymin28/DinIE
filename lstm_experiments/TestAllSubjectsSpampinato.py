
import subprocess

# subprocess.run(["conda", "activate", "dinov2"])


def run_script(gallery_subject, query_subject, query_gallery):
    command = [
        "python",
        "LSTMDistillRetreivalSpampinato.py",
        f"--gallery_subject={gallery_subject}",
        f"--query_subject={query_subject}",
        f"--query_gallery={query_gallery}"
    ]
    subprocess.run(command)

# Define your parameters
gallery_subjects = [1,2,3,4,5,6]
query_subject = 4  # Assuming this is constant for all queries
query_galleries = ["test", "val"]

# Loop through the combinations
for gallery_subject in gallery_subjects:
    query_subject  = gallery_subject
    for query_gallery in query_galleries:
        run_script(gallery_subject, query_subject, query_gallery)