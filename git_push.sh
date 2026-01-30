echo "push the update to github"

git add .
git commit -m "debug"
git branch -M main
git push -u origin main
