(ns gec.prepare
  (:require [clojure.java.io :as io]
            [clojure.java.shell :as sh]
            [clojure.set :as set]
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

(def append-newline
  (partial (aid/flip str) "\n"))

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
                  "10000"
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

(defn build-vocabulary
  [dataset]
  (with-open [bpe-file (io/reader (get-dataset-path dataset "bpe.txt"))]
    (->> bpe-file
         line-seq
         (reduce (fn [reduction sentence]
                   (reduce (fn [reduction* word]
                             (conj reduction* word))
                           reduction
                           (str/split sentence #" ")))
                 #{})
         (map-indexed (fn [index word]
                        ;consider EOS and SOS tokens
                        {(+ 2 index) word}))
         (apply merge {0 "<SOS>"
                       1 "<EOS>"})
         ;TODO don't spit index.json
         ((juxt (comp (partial helpers/spit-parents
                               (get-dataset-path dataset "word.json"))
                      generate-string)
                (comp (partial helpers/spit-parents
                               (get-dataset-path dataset "index.json"))
                      generate-string
                      set/map-invert))))))

;(def bag
;  (comp (partial s/transform* s/MAP-VALS count)
;        (partial group-by int)))

(defn bag
  [s]
  (reduce (fn [reduction c]
            (if (< (int c) 128)
              (s/transform (s/nthpath (int c)) inc reduction)
              reduction))
          (repeat 128 0) s))

(def split-tokens
  (partial (aid/flip str/split) #" "))

(defn bag-validation
  [process sentences]
  (if process
    (map (comp generate-string
               (partial map bag)
               split-tokens)
         sentences)
    sentences))

(defn structure-sentence
  [sentence]
  {:word   sentence
   :length (->> sentence
                split-tokens
                count)
   :bag    (->> sentence
                split-tokens
                (map bag))})

(def get-count-filename
  (comp (partial (aid/flip str) ".txt")
        count))

(defn make-split*
  [training]
  (fn [dataset n]
    (let [index (parse-string (slurp (get-dataset-path dataset "index.json")))]
      (with-open [random-file (io/reader (get-dataset-path dataset "random.txt"))]
        (with-open [bpe-file (io/reader (get-dataset-path dataset "bpe.txt"))]
          (->> random-file
               line-seq
               ((if training
                  drop
                  take) n)
               (map structure-sentence)
               (map (fn [bpe m]
                      (s/setval :bpe bpe m))
                    (->> bpe-file
                         line-seq
                         ((if training
                            drop
                            take) n)
                         (map (comp (partial map index)
                                    split-tokens))))
               (map
                 (fn [m]
                   (helpers/spit-parents
                     (get-dataset-path dataset
                                       "split"
                                       (if training
                                         "training"
                                         "validation")
                                       (get-count-filename (:bpe m)))
                     (append-newline (generate-string m))
                     :append
                     true)))
               dorun))))))

(def split
  (apply juxt (map make-split* [false true])))
