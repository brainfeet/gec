(ns gec.helpers
  (:require [clojure.java.shell :as sh]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [aid.core :as aid]
            [cheshire.core :refer :all]
            [gec.command :as command]
            [me.raynes.fs :as fs]))

(def join-paths
  (comp str
        io/file))

(def parse-keywordize
  (partial (aid/flip parse-string) true))

(defn spit-parents
  [f & more]
  (-> f
      fs/parent
      fs/mkdirs)
  (apply spit f more))

(def get-repository
  (comp last
        (partial (aid/flip str/split) #"/")))

(defn clone-checkout
  [directory url commit]
  (aid/mlet [_ (->> (command/git "clone" url)
                    (sh/with-sh-dir directory))]
            (sh/with-sh-dir (io/file directory (get-repository url))
                            (command/git "checkout" commit))))

(defn python
  [& more]
  (sh/with-sh-dir "python"
                  (apply command/export
                         "PYTHONPATH=$(pwd)"
                         "&&"
                         "source"
                         "activate"
                         "gec"
                         "&&"
                         "python"
                         more)))
