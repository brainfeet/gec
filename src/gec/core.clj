(ns gec.core
  (:gen-class)
  (:require [gec.task.rsync :as rsync]))

(defn -main
  [command & more]
  (apply (comp println
               ({"rsync" rsync/rsync}
                 command))
         more)
  ;rsync/rsync doesn't exit immediately
  (shutdown-agents))
