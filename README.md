# al-folio

## Commands to modify/run 

To get a local representation of how the website should look:

`bundle exec jekyll serve` 

This will create a localhost with which to dynamically interact with the website. It is better to use this version when modifying intensively as changes are syncronized instantly without the need for pushing/committing.

To deploy:

1. Commit and push all your changes
2. `bin/deploy --user`   

It might take several minutes for this to finish and the updates on the actual website might come out after a while.  
You should see the following output in your terminal.

```
Deploying...
Source branch: master
Deploy branch: gh-pages
Do you want to proceed? [y/N] y
...
...
...
Switched to branch 'master'
Your branch is up to date with 'origin/master'.
Deployed successfully!

```


