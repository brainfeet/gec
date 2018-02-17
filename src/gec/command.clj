(ns gec.command
  ;TODO remove io
  (:require [clojure.java.io :as io]
            [clojure.java.shell :as sh]
            [clojure.string :as str]
            [aid.core :as aid]
            [cats.monad.either :as either]
            [com.rpl.specter :as s]
            [me.raynes.fs :as fs]))

(defn if-then-else
  ;TODO move this function to aid
  [if-function then-function else]
  ((aid/build if
              if-function
              then-function
              identity)
    else))

(def err
  #"stty: 'standard input': Inappropriate ioctl for device\n")

(defn execute
  [shell & more]
  (->> more
       (concat ["source" (fs/expand-home (str "~/." shell "rc")) "&&"])
       (str/join " ")
       (sh/sh shell "-c")
       (s/transform :err (fn [s]
                           (str/replace s err "")))
       ((partial if-then-else
                 (comp empty?
                       :err)
                 (partial (aid/flip dissoc) :err)))))

(def escape
  (partial if-then-else
           (partial re-find #"^\d")
           (partial str "*")))

(def out
  (comp str/trim-newline
        :out))

(defn monadify
  [shell command]
  (comp (aid/build if
                   (comp zero?
                         :exit)
                   (comp either/right
                         out)
                   (comp either/left
                         (juxt out
                               (comp str/trim-newline
                                     :err))))
        (partial execute shell command)))

(defn make-defcommand
  [shell]
  (fn [command]
    (eval `(def ~(symbol (escape command))
             (monadify ~shell ~command)))))

(defn defcommands
  [shell]
  (->> "compgen -bc"
       (execute shell)
       :out
       str/split-lines
       set
       (run! (make-defcommand shell))))

(def ent
  (->> "user.name"
       System/getProperty
       (sh/sh "getent" "passwd")))

(def parse-ent
  (comp second
        (partial re-find #"\/([^/]+)\n")
        :out))

(defcommands (parse-ent ent))
