(define (domain gripper-strips)
   (:predicates (room ?r)
		(ball ?b)
		(gripper ?g)
		(at-robby ?r)
		(at ?b ?r)
		(free ?g)
		(carry ?o ?g)
        (red ?o)
        (blue ?o)
        )

   (:action move
       :parameters  (?from ?to)
       :precondition (and  (room ?from) (room ?to) (at-robby ?from))
       :effect (and  (at-robby ?to)
		     (not (at-robby ?from)))
       :cost (+ 
              (sum (?room ?ball) (and (at ?ball ?room) (red ?ball) (blue ?room)))
              (sum (?room ?ball) (and (at ?ball ?room) (blue ?ball) (red ?room)))
           )
       
       )

   (:action pick
       :parameters (?obj ?room ?gripper)
       :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
			    (at ?obj ?room) (at-robby ?room) (free ?gripper))
       :effect (and (carry ?obj ?gripper)
		    (not (at ?obj ?room)) 
		    (not (free ?gripper)))
       :cost 0
   )


   (:action drop
       :parameters  (?obj  ?room ?gripper)
       :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
			    (carry ?obj ?gripper) (at-robby ?room))
       :effect (and (at ?obj ?room)
		    (free ?gripper)
		    (not (carry ?obj ?gripper)))
       :cost 0
   )

   ;(:action color-blue
   ; :parameters (?b)
   ; :precondition (and (ball ?b) (red ?b))
   ; :effect (and (blue ?b) (not (red ?b)))
   ; :cost 0
   ;)

   ;(:action color-red
   ; :parameters (?b)
   ; :precondition (and (ball ?b) (blue ?b))
   ; :effect (and (red ?b) (not (blue ?b)))
   ; :cost 0
   ;)
)

