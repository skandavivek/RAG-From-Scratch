#init
git init
heroku git:remote -a rag-basic

git add .
git commit -m "Initial commit"
git push heroku master

#scaling
heroku ps:scale web=1

heroku config:set WEB_CONCURRENCY=1
heroku config:set TIMEOUT=3600

#redis
heroku addons:create heroku-redis:mini
heroku ps:scale web=1 worker=1