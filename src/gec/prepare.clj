(ns gec.prepare
  (:require [clojure.java.io :as io]
            [clojure.java.shell :as sh]
            [clojure.string :as str]
            [aid.core :as aid]
            [cats.monad.either :as either]
            [cheshire.core :refer :all]
            [com.rpl.specter :as s]
            [me.raynes.fs :as fs]
            [gec.command :as command]
            [gec.helpers :as helpers]))

(def get-dataset-path
  (partial helpers/join-paths "resources/dataset"))

(def parse
  (comp (partial helpers/python
                 "gec/parse.py"
                 "--path")
        fs/absolute
        (partial (aid/flip get-dataset-path) "isolated")))

(defn make-find-files
  [directory-name re]
  (comp (partial (aid/flip fs/find-files) re)
        (partial (aid/flip get-dataset-path) directory-name)))

(def find-parsed-files
  (make-find-files "parsed" #"\d+\.json"))

(def get-parsed-texts
  (comp (partial map slurp)
        find-parsed-files))

(def parse-keywordize
  (partial (aid/flip parse-string) true))

(def split-sentences
  (comp (partial map flatten)
        (partial partition 2)
        (partial s/setval* s/BEGINNING [[]])
        (partial partition-by :is_sent_start)))

(def is-ascii?
  (partial every? (comp (partial > 128)
                        int)))

(def has-newline?
  (partial re-find #".*\n.*"))

(defn append-newline
  [s]
  (str s "\n"))

(def get-plain
  (comp (partial map (comp append-newline
                           (partial str/join " ")
                           (partial remove has-newline?)
                           (partial filter is-ascii?)
                           (partial map :text)))))

(def convert
  (comp get-plain
        split-sentences
        parse-keywordize))

(defn make-append
  [dataset]
  (fn [sentence]
    (spit (get-dataset-path dataset "combined.txt") sentence :append true)))

(def combine
  (comp dorun
        (aid/build map
                   make-append
                   (comp (partial mapcat convert)
                         get-parsed-texts))))

(def randomize
  (aid/build (partial command/shuf "-o")
             (partial (aid/flip get-dataset-path) "random.txt")
             (partial (aid/flip get-dataset-path) "combined.txt")))

(defn learn-bpe
  [dataset]
  (command/python "bin/learn_bpe.py"
                  "-s"
                  "30000"
                  "<"
                  (get-dataset-path dataset "random.txt")
                  ">"
                  (get-dataset-path dataset "codes.txt")))

(defn apply-bpe
  [dataset]
  (command/python "bin/apply_bpe.py"
                  "-c"
                  (get-dataset-path dataset "codes.txt")
                  "<"
                  (get-dataset-path dataset "random.txt")
                  ">"
                  (get-dataset-path dataset "bpe.txt")))

(defn make-split-validation*
  [dataset n]
  (fn [combined split]
    (with-open [file (io/reader (get-dataset-path dataset combined))]
      (helpers/spit-parents (get-dataset-path dataset "validation" split)
                            (str/join "\n" (take n (line-seq file)))))))

(defn split-validation
  [dataset n]
  (map (make-split-validation* dataset n)
       ["random.txt" "bpe.txt"]
       ["input.txt" "output.txt"]))

(defn make-split-training*
  [dataset n]
  (fn [combined split]
    (with-open [file (io/reader (get-dataset-path dataset combined))]
      (dorun (map (fn [sentence]
                    (helpers/spit-parents (get-dataset-path dataset
                                                            "training"
                                                            split
                                                            (str (count (str/split sentence #" ")) ".txt"))
                                          (str sentence "\n")
                                          :append
                                          true))
                  (drop n (line-seq file)))))))

(defn split-training
  [dataset n]
  (map (make-split-training* dataset n)
       ["random.txt" "bpe.txt"]
       ["input" "output"]))
