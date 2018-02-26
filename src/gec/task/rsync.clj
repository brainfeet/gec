(ns gec.task.rsync
  (:require [gec.command :as command]))

(def rsync
  (partial command/rsync
           "-azP"
           "--include=resources/hyperparameter/*"
           "--exclude=.idea"
           ;.gitignore in the project root seems to be used as a filter by default
           "--filter=':- /python/.gitignore'"
           (System/getProperty "user.dir")))
